city = {
    "A" : (-150, 0),
    "B" : (0, 100),
    "C" : (0, -100),
    "D" : (150, 0),
    "E" : (0,0)
}
roads = [
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("E", "A"),
]
for start, end in roads:
    print (start, "connects to", end)
travel_times = {
    ("A", "B"): 4,
    ("B", "C"): 5,
    ("C", "D"): 3,
    ("D", "E"): 4,
    ("E", "A"): 6
}
route1 = (
    travel_times[("A", "B")]
    + travel_times[("B", "C")]
    + travel_times[("C", "D")]
)
route2 = (
    travel_times[("E", "A")]
    + travel_times[("D", "E")]
)

# Create an empty graph
graph = {}

# Add every intersection
for intersection in city:
    graph[intersection] = []

# Add roads and travel times
for (start, end), minutes in travel_times.items():
    graph[start].append((end, minutes))
    graph[end].append((start, minutes))

print(graph["A"])
distances = {}

for intersection in city:
    distances[intersection] = float("inf")
distances["A"] = 0
current = "A"

for neighbor, minutes in graph[current]:
    new_distance = distances[current] + minutes

    if new_distance < distances[neighbor]:
        distances[neighbor] = new_distance
print(distances)
print("Route 1 takes:", route1, "minutes")
print("Route 2 takes:", route2, "minutes")
if route1 < route2:
    print("Route 1 is faster")

elif route2 < route1:
    print("Route 2 is faster")

else:
    print("Both routes take the same time")

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


# Draw all intersections
for name, (x, y) in city.items():
    pen.penup()
    pen.goto(x, y)
    pen.dot(40, "skyblue")

    pen.goto(x, y - 10)
    pen.color("black")
    pen.write(name, align="center",
              font=("Arial", 12, "bold"))
Screen.mainloop()

