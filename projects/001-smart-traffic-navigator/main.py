
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


def build_graph():
    graph = {}

    for intersection in city:
        graph[intersection] = []

    for (start, end), minutes in travel_times.items():
        graph[start].append((end, minutes))
        graph[end].append((start, minutes))

    return graph


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

        if current == destination:
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


screen = turtle.Screen()
screen.setup(700, 450)
screen.bgcolor("#101827")
screen.title("Smart Traffic Navigator")
screen.tracer(0)

pen = turtle.Turtle()
pen.speed(0)
pen.hideturtle()

car = turtle.Turtle()
car.shape("circle")
car.color("yellow")
car.shapesize(0.8)
car.penup()
car.speed(0)

traffic_active = False

car_state = {
    "route": [],
    "segment": 0,
    "progress": 0.0
}


def draw_city():
    graph = build_graph()

    route, best_time = find_fastest_route(graph, "A", "D")

    car_state["route"] = route
    car_state["segment"] = 0
    car_state["progress"] = 0.0

    if route:
        car.showturtle()
        car.goto(city[route[0]])
    else:
        car.hideturtle()

    pen.clear()
    pen.pensize(7)

    for start, end in roads:
        pen.penup()
        pen.goto(city[start])

        if traffic_active and (start, end) == ("E", "A"):
            pen.color("red")
        else:
            pen.color("gray")

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

    pen.penup()
    pen.goto(0, 175)
    pen.color("white")

    if traffic_active:
        status = "TRAFFIC JAM ACTIVE"
    else:
        status = "NORMAL TRAFFIC"

    pen.write(
        status,
        align="center",
        font=("Arial", 17, "bold")
    )

    pen.goto(0, -170)

    pen.write(
        f"Fastest Route: {' -> '.join(route)}",
        align="center",
        font=("Arial", 13, "bold")
    )

    pen.goto(0, -195)

    pen.write(
        f"Travel Time: {best_time} minutes | Press T to toggle traffic",
        align="center",
        font=("Arial", 11, "normal")
    )

    screen.update()

    print("Traffic status:", status)
    print("Fastest route:", " -> ".join(route))
    print("Travel time:", best_time, "minutes")
    print("----------------------------")


def toggle_traffic():
    global traffic_active

    traffic_active = not traffic_active

    if traffic_active:
        travel_times[("E", "A")] = 15
    else:
        travel_times[("E", "A")] = 6

    draw_city()


def move_car():
    route = car_state["route"]
    segment = car_state["segment"]

    if segment < len(route) - 1:
        start = route[segment]
        end = route[segment + 1]

        x1, y1 = city[start]
        x2, y2 = city[end]

        progress = min(1.0, car_state["progress"] + 0.02)

        x = x1 + (x2 - x1) * progress
        y = y1 + (y2 - y1) * progress

        car.goto(x, y)

        if progress >= 1.0:
            car_state["segment"] += 1
            car_state["progress"] = 0.0
        else:
            car_state["progress"] = progress

    screen.update()
    screen.ontimer(move_car, 40)


draw_city()

screen.onkey(toggle_traffic, "t")
screen.onkey(toggle_traffic, "T")
screen.listen()

move_car()

screen.mainloop()
