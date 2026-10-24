
import turtle

city = {
    "A": (-150, 0),
    "B": (0, 100),
    "C": (0, -100),
    "D": (150, 0),
    "E": (0, 0)
}

roads = [
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("E", "A")
]

travel_times = {
    ("A", "B"): 4,
    ("B", "C"): 5,
    ("C", "D"): 3,
    ("D", "E"): 4,
    ("E", "A"): 6
}

traffic = input("Simulate traffic jam? (yes/no): ").strip().lower()

if traffic == "yes":
    travel_times[("E", "A")] = 15
    print("Traffic jam activated!")
else:
    print("Normal traffic conditions.")

graph = {}

for intersection in city:
    graph[intersection] = []

for (start, end), minutes in travel_times.items():
    graph[start].append((end, minutes))
    graph[end].append((start, minutes))


def find_fastest_route(graph, source, destination):
    distances = {}
    previous = {}
    visited = set()

    for intersection in graph:
        distances[intersection] = float("inf")
        previous[intersection] = None

    distances[source] = 0

    while len(visited) < len(graph):
        current = None
        smallest_time = float("inf")

        for intersection in graph:
            if intersection not in visited and distances[intersection] < smallest_time:
                smallest_time = distances[intersection]
                current = intersection

        if current is None:
            break

        visited.add(current)

        for neighbor, minutes in graph[current]:
            if neighbor not in visited:
                new_distance = distances[current] + minutes

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current

    route = []

    if distances[destination] == float("inf"):
        return [], float("inf")

    current = destination

    while current is not None:
        route.insert(0, current)
        current = previous[current]

    return route, distances[destination]


route, best_time = find_fastest_route(graph, "A", "D")

print("Fastest route:", " -> ".join(route))
print("Total time:", best_time, "minutes")

screen = turtle.Screen()
screen.setup(700, 450)
screen.bgcolor("#B21020")
screen.title("Smart Traffic Navigator")

pen = turtle.Turtle()
pen.speed(0)
pen.hideturtle()

pen.pensize(7)
pen.color("gray")

for start, end in roads:
    pen.penup()
    pen.goto(city[start])
    pen.pendown()
    pen.goto(city[end])

pen.color("lime")
pen.pensize(11)

for i in range(len(route) - 1):
    start = route[i]
    end = route[i + 1]

    pen.penup()
    pen.goto(city[start])
    pen.pendown()
    pen.goto(city[end])

for name, (x, y) in city.items():
    pen.penup()
    pen.goto(x, y)
    pen.dot(40, "skyblue")

    pen.goto(x, y - 10)
    pen.color("black")
    pen.write(
        name,
        align="center",
        font=("Arial", 12, "bold")
    )

screen.mainloop()
