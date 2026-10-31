# 100 Days, 100 Projects

> A public project journal by [Keshav Markan](https://keshavmarkan.com)—exploring algorithms, applied AI, computer vision, and software engineering through working projects.

[Explore the portfolio](https://keshavmarkan.com/#folio-series) · [Follow @kaylocksin](https://www.instagram.com/kaylocksin/) · [GitHub profile](https://github.com/marrkann08-max)

## Project index

| Day | Project | What it explores | Code / demo |
| :--- | :--- | :--- | :--- |
| 001 | **Smart Traffic Navigator** | Dijkstra’s shortest path and changing traffic conditions | [Source](projects/001-smart-traffic-navigator) |
| 002 | **Interactive portfolio** | React, TypeScript, an explorable Three.js campus, and Cloudflare Workers + D1 | [Live website](https://keshavmarkan.com) |
| 003 | **Ki Forge** | MediaPipe hand tracking, an OpenCV/NumPy particle renderer, and synthesized sound | [Setup & source](projects/003-ki-forge) |

Day 002 is hosted separately; this repository contains the source folders listed above. New entries are added as projects are published.

## Latest experiment — Ki Forge

Open a palm to charge an energy ball. Bring two palms together to shape it, then close your fist to compress and burst the particles with trails and shockwaves.

![Ki Forge particle burst](projects/003-ki-forge/assets/blast-preview.png)

**Python · OpenCV · MediaPipe · NumPy · sounddevice**

Hand tracking runs locally after the model download. A mouse playground lets you try the renderer without a webcam.

[Run Ki Forge →](projects/003-ki-forge/README.md)

## Repository layout

```text
projects/
├── 001-smart-traffic-navigator/   shortest-path traffic simulation
└── 003-ki-forge/                 gesture-controlled energy playground
```

## Running a project

Each project has its own dependencies and entry point. Open its folder and follow the available setup instructions. Ki Forge includes a dedicated README and automated checks; the traffic navigator includes its Python source and experiment outputs.

## About the series

The goal is to connect concepts from class and independent learning with something you can see, use, or test. Project scope varies; the journal documents experiments as well as larger applications.

See the visual trail at **[keshavmarkan.com](https://keshavmarkan.com/#folio-series)**.
