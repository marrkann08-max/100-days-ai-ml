# 100 Days, 100 Projects

I'm [Keshav](https://keshavmarkan.com), a computer science student at Conestoga College. I'm building a project a day and sharing what I learn.

[Explore the portfolio](https://keshavmarkan.com/#folio-series) · [Follow @kaylocksin](https://www.instagram.com/kaylocksin/) · [GitHub profile](https://github.com/marrkann08-max)

## Project index

| Day | Project | What it explores | Code / demo |
| :--- | :--- | :--- | :--- |
| 001 | **Smart Traffic Navigator** | Dijkstra’s shortest path and changing traffic conditions | [Source](projects/001-smart-traffic-navigator) |
| 002 | **Interactive portfolio** | React, TypeScript, an explorable Three.js campus, and Cloudflare Workers + D1 | [Website](https://keshavmarkan.com) · [Source](https://github.com/marrkann08-max/keshav-portfolio) |
| 003 | **Ki Forge** | MediaPipe hand tracking, an OpenCV/NumPy particle renderer, and synthesized sound | [Setup & source](projects/003-ki-forge) |
| 004 | **Opcode Canvas** | Editable x86 assembly, a Python pixel canvas, and instruction stepping | [Setup & source](projects/004-opcode-canvas) |
| 005 | **Descent** | A *Platform*-inspired tower where NumPy neural-net brains evolve greed or sharing via a genetic algorithm | [Setup & source](projects/005-descent) |
| 006 | **Holo Window** | Webcam head tracking and off-axis projection turn a laptop screen into a 3D window; Depth Anything turns photos into scenes | [Setup & source](projects/006-holo-window) |
| 007 | **Primordial** | GPU particle life where a genetic algorithm discovers life-like species on its own | [Setup & source](projects/007-primordial) |
| 008 | **Federated Traffic Forecasting** | Graph + recurrent networks on METR-LA trained across districts with Flower: benchmarks, communication trade-offs, differential privacy, real deployment, paper | [Report & source](projects/008-fedtraffic) |

The portfolio source lives in [keshav-portfolio](https://github.com/marrkann08-max/keshav-portfolio). The other projects are in this repository.

## Latest experiment — Federated Traffic Forecasting

A small research project: can four city districts train one traffic-forecasting model without sharing their sensor data? Graph convolutions and GRU/LSTM encoders written from scratch, federated learning with Flower (simulated and as a real multi-process deployment), 40 logged and reproducible runs, and a paper-style write-up.

![Federated districts on the METR-LA road graph](projects/008-fedtraffic/results/figures/eda_sensor_map.png)

**Python · PyTorch · Flower · graph neural networks · GRU/LSTM · federated learning · differential privacy**

The road graph cuts 60-minute error by 6%. FedAvg comes within 2.8% of centralised training while keeping raw data local, 8-bit updates cut communication by 37% for free, and districts differ so much that each training alone does best.

[Read the report →](projects/008-fedtraffic/paper/report.md)

## Repository layout

```text
projects/
├── 001-smart-traffic-navigator/   shortest-path traffic simulation
├── 003-ki-forge/                 gesture-controlled energy playground
├── 004-opcode-canvas/            assembly-driven pixel canvas
├── 005-descent/                  evolving greed vs. sharing in a descending-feast tower
├── 006-holo-window/              head-tracked 3D window on a laptop screen
├── 007-primordial/               particle life with evolved species
└── 008-fedtraffic/               federated spatio-temporal traffic forecasting (research project)
```

## Running a project

Each project has its own dependencies and entry point. Open its folder and follow the setup instructions. Ki Forge, Opcode Canvas, Descent, Holo Window, Primordial and Federated Traffic Forecasting include dedicated READMEs and automated checks; the traffic navigator includes its Python source and experiment outputs.

## About the series

Some projects start with a concept from class; others start with an idea I want to try. Each entry links to the code and any setup instructions.

See the visual trail at **[keshavmarkan.com](https://keshavmarkan.com/#folio-series)**.
