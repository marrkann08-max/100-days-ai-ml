from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    caption: str
    source: str


PRESETS = (
    Preset("XOR tide", "Try changing xor to and, then press Ctrl+Enter.", """; eax = x, ebx = y, ecx = time
; return a colour index in eax

xor eax, ebx
add eax, ecx
shl eax, 2
ret
"""),
    Preset("Interference", "Two moving integer waves, folded into eight bits.", """; eax = x, ebx = y, ecx = time
; multiply, shift, fold

add eax, ecx
imul eax, eax
sub ebx, ecx
imul ebx, ebx
add eax, ebx
shr eax, 4
ret
"""),
    Preset("Bit garden", "Change the shift amounts to reshape the repeating cells.", """; eax = x, ebx = y, ecx = time
; bit operations become geometry

mov edx, eax
shr edx, 3
add ebx, ecx
xor eax, ebx
imul eax, edx
shr ebx, 2
xor eax, ebx
ret
"""),
)
