"""Assemble x86-32 code and run it against a mapped framebuffer.

The Python renderer only applies a palette. Every colour index is computed
by the user program, including during exports and instruction stepping.
"""
from dataclasses import dataclass
import time

from keystone import Ks, KsError, KS_ARCH_X86, KS_MODE_32
from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from unicorn import Uc, UcError, UC_ARCH_X86, UC_MODE_32
from unicorn.x86_const import (
    UC_X86_REG_EAX, UC_X86_REG_EBX, UC_X86_REG_ECX, UC_X86_REG_EDX,
    UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_ESP,
    UC_X86_REG_EIP, UC_X86_REG_EFLAGS,
)

CODE = 0x100000
BUFFER = 0x200000
STACK = 0x300000
PAGE = 0x1000
WIDTH, HEIGHT = 128, 80
TIMEOUT_US = 60_000
INSTRUCTION_LIMIT = 2_000_000
REGISTERS = {"EAX": UC_X86_REG_EAX, "EBX": UC_X86_REG_EBX,
             "ECX": UC_X86_REG_ECX, "EDX": UC_X86_REG_EDX,
             "ESI": UC_X86_REG_ESI, "EDI": UC_X86_REG_EDI,
             "ESP": UC_X86_REG_ESP, "EIP": UC_X86_REG_EIP}


class AssemblyError(ValueError):
    pass


class ExecutionError(RuntimeError):
    pass


def assemble(source: str) -> bytes:
    # Strip semicolon comments: Keystone treats semicolons as separators.
    clean = "\n".join(line.split(";", 1)[0] for line in source.splitlines())
    if not clean.strip():
        raise AssemblyError("Write a program first. End it with ret.")
    if len(source) > 12_000:
        raise AssemblyError("Keep the program under 12,000 characters.")
    try:
        encoded, _ = Ks(KS_ARCH_X86, KS_MODE_32).asm(clean, CODE)
    except KsError as exc:
        raise AssemblyError(f"Cannot assemble: {exc}") from exc
    if not encoded or len(encoded) > PAGE:
        raise AssemblyError("Program must contain 1–4096 bytes of instructions.")
    return bytes(encoded)


def machine(program: bytes, width: int, height: int) -> Uc:
    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    uc.mem_map(CODE, PAGE * 4)
    uc.mem_write(CODE, program)
    uc.mem_map(BUFFER, ((width * height + PAGE - 1) // PAGE) * PAGE)
    uc.mem_map(STACK, PAGE * 4)
    uc.reg_write(UC_X86_REG_ESP, STACK + PAGE * 4 - 16)
    return uc


@dataclass(frozen=True)
class Frame:
    pixels: bytes
    milliseconds: float
    tick: int


class PixelProgram:
    def __init__(self, source: str, width: int = WIDTH, height: int = HEIGHT):
        if not (1 <= width <= 256 and 1 <= height <= 256):
            raise ValueError("Framebuffer dimensions must be between 1 and 256.")
        self.source = source
        self.shader = assemble(source)
        self.width, self.height = width, height
        # The shader is at CODE; its relative branches keep their addresses.
        # The driver follows on the next page and calls it once per pixel.
        self.driver_address = CODE + PAGE

    def render(self, tick: int) -> Frame:
        driver = f"""
            mov edi, {BUFFER}
            xor esi, esi
        pixel_loop:
            mov eax, esi
            xor edx, edx
            mov ebx, {self.width}
            div ebx
            mov ebx, eax
            mov eax, edx
            mov ecx, {tick & 0xffffffff}
            xor edx, edx
            push esi
            push edi
            call {CODE}
            pop edi
            pop esi
            cld
            stosb
            inc esi
            cmp esi, {self.width * self.height}
            jb pixel_loop
        """
        encoded, _ = Ks(KS_ARCH_X86, KS_MODE_32).asm(driver, self.driver_address)
        uc = machine(self.shader, self.width, self.height)
        uc.mem_write(self.driver_address, bytes(encoded))
        end = self.driver_address + len(encoded)
        start = time.perf_counter()
        try:
            uc.emu_start(self.driver_address, end, timeout=TIMEOUT_US,
                         count=INSTRUCTION_LIMIT)
        except UcError as exc:
            raise ExecutionError(f"CPU stopped: {exc}. Check your code and ret.") from exc
        if uc.reg_read(UC_X86_REG_EIP) != end:
            raise ExecutionError("Execution limit reached. Check for a loop or missing ret.")
        elapsed = (time.perf_counter() - start) * 1000
        return Frame(bytes(uc.mem_read(BUFFER, self.width * self.height)), elapsed, tick)

    def debugger(self, x: int, y: int, tick: int):
        return PixelDebugger(self, x, y, tick)


class PixelDebugger:
    def __init__(self, program: PixelProgram, x: int, y: int, tick: int):
        self.uc = machine(program.shader, program.width, program.height)
        self.end = CODE + PAGE * 3
        stack = self.uc.reg_read(UC_X86_REG_ESP) - 4
        self.uc.mem_write(stack, self.end.to_bytes(4, "little"))
        self.uc.reg_write(UC_X86_REG_ESP, stack)
        self.uc.reg_write(UC_X86_REG_EAX, x)
        self.uc.reg_write(UC_X86_REG_EBX, y)
        self.uc.reg_write(UC_X86_REG_ECX, tick & 0xffffffff)
        self.uc.reg_write(UC_X86_REG_EDX, 0)
        # Match the driver at shader entry, including saved registers.
        self.uc.reg_write(UC_X86_REG_ESI, y * program.width + x)
        self.uc.reg_write(UC_X86_REG_EDI, BUFFER + y * program.width + x)
        self.uc.reg_write(UC_X86_REG_EIP, CODE)
        self.steps = 0
        self.done = False

    def snapshot(self):
        values = {name: self.uc.reg_read(reg) for name, reg in REGISTERS.items()}
        values["FLAGS"] = self.uc.reg_read(UC_X86_REG_EFLAGS)
        return values

    def instruction(self):
        if self.done:
            return "returned"
        address = self.uc.reg_read(UC_X86_REG_EIP)
        try:
            raw = bytes(self.uc.mem_read(address, 15))
            decoded = next(Cs(CS_ARCH_X86, CS_MODE_32).disasm(raw, address, count=1), None)
        except UcError:
            return "unmapped address"
        if decoded:
            return f"{decoded.mnemonic} {decoded.op_str}".strip()
        return "invalid instruction"

    def step(self):
        if self.done:
            return self.snapshot()
        if self.steps >= 512:
            raise ExecutionError("Debugger limit reached (512 steps). Reset or edit the loop.")
        try:
            self.uc.emu_start(self.uc.reg_read(UC_X86_REG_EIP), self.end,
                              timeout=TIMEOUT_US, count=1)
        except UcError as exc:
            raise ExecutionError(f"CPU stopped: {exc}") from exc
        self.steps += 1
        self.done = self.uc.reg_read(UC_X86_REG_EIP) == self.end
        return self.snapshot()
