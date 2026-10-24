city = {
    "A" : (-150, 0),
    "B" : (0, 100),
    "C" : (0, -100),
    "D" : (150, 0)
}
roads = [
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "A"),
]
for start, end in roads:
    print (start, "connects to", end)


import turtle
Screen = turtle.Screen()
Screen.setup(700, 450)
Screen.bgcolor("#B21020")
Screen.title("Smart Traffic Navigator")

pen = turtle.Turtle()
pen.speed(0)
pen.pensize(7)
pen.color("Gray")
pen.hideturtle()
for start, end in roads:
    pen.penup()
    pen.goto(city[start])
    pen.pendown()
    pen.goto(city[end])

Screen.mainloop()
