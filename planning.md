# Project Plan

> Status: draft template. Fill in the `TBD` fields as decisions are made.

## 0. Overview

- **Project goal:** TBD (one or two sentences on what we are building and why)
- **Definition of done:** TBD (e.g. working agent pipeline, trained model, report, demo)
- **Deadline:** TBD
- **Constraints:** TBD (language/framework, compute, budget, data licensing)
- **Team / owners:** TBD

---

## 1. Research

**Goal:** frame the problem clearly and decide what we need to learn before building.

### 1.1 Problem framing
- Problem statement: TBD
- Inputs: TBD
- Expected outputs: TBD
- Who uses the result and how: TBD

### 1.2 Research questions
- [ ] RQ1: TBD
- [ ] RQ2: TBD
- [ ] RQ3: TBD

### 1.3 Prior work and references
| Source | Type (paper / repo / article) | Key takeaway | Link |
|--------|-------------------------------|--------------|------|
| TBD    |                               |              |      |

### 1.4 Deliverable
- [ ] Short summary of findings and the chosen approach

---

## 2. Datasets

**Goal:** find candidate datasets, compare them and pick a shortlist.

### 2.1 Where to search
- [ ] Kaggle
- [ ] Hugging Face Datasets
- [ ] Google Dataset Search
- [ ] UCI Machine Learning Repository
- [ ] Papers with Code / academic papers
- [ ] Government / open data portals
- [ ] Public APIs relevant to the domain
- [ ] Other: TBD

### 2.2 Evaluation criteria
- **Relevance:** does it match the problem and research questions?
- **Size:** enough rows/samples for the approach?
- **Quality:** missing values, noise, label accuracy
- **Format and access:** CSV, JSON, API, database; download or live access
- **Licence:** can we use it for this project?
- **Freshness:** how recent, how often updated
- **Documentation:** schema, data dictionary, provenance

### 2.3 Candidate shortlist
| Dataset | Source | Size | Format | Licence | Relevance (1–5) | Quality (1–5) | Notes |
|---------|--------|------|--------|---------|-----------------|---------------|-------|
| TBD     |        |      |        |         |                 |               |       |

### 2.4 Deliverable
- [ ] Final selected dataset(s) with justification

---

## 3. Data Pipeline

**Goal:** load, clean and prepare the data so agents and tools can use it.

### 3.1 Ingestion
- Source(s) and access method: TBD
- Storage location (local files / database / cloud): TBD

### 3.2 Cleaning and preparation
- [ ] Handle missing values
- [ ] Remove duplicates and outliers
- [ ] Normalise formats (dates, units, text encoding)
- [ ] Feature engineering / enrichment: TBD
- [ ] Train / validation / test split (if applicable)

### 3.3 Exploratory data analysis
- [ ] Basic statistics and distributions
- [ ] Key visualisations
- [ ] Notes on bias or gaps in the data

### 3.4 Deliverable
- [ ] Reproducible pipeline script or notebook
- [ ] Clean dataset ready for the agents

---

## 4. Agent and Tool Architecture

**Goal:** define the agents, their tools and how they connect.

### 4.1 Agents
| Agent | Responsibility | Inputs | Outputs | Tools used |
|-------|----------------|--------|---------|------------|
| TBD (e.g. Research agent) |  |  |  |  |
| TBD (e.g. Data agent)     |  |  |  |  |
| TBD (e.g. Analysis agent) |  |  |  |  |
| TBD (e.g. Orchestrator)   |  |  |  |  |

### 4.2 Tools
| Tool | Purpose | Implementation (API / function / MCP server) | Used by |
|------|---------|-----------------------------------------------|---------|
| TBD (e.g. web search) |  |  |  |
| TBD (e.g. dataset loader) |  |  |  |
| TBD (e.g. database query) |  |  |  |

### 4.3 Orchestration and data flow
- Orchestration pattern (sequential / orchestrator–workers / parallel): TBD
- How agents pass results to each other: TBD
- Error handling and retries: TBD
- Human-in-the-loop checkpoints: TBD

```
[Diagram placeholder]
User request -> Orchestrator -> Agents -> Tools -> Data -> Output
```

### 4.4 Tech stack
- Language: TBD
- Agent framework / SDK: TBD
- Models: TBD
- Storage: TBD

---

## 5. Build and Evaluation

**Goal:** build in milestones and check results at each step.

### 5.1 Milestones
| # | Milestone | Description | Target date | Status |
|---|-----------|-------------|-------------|--------|
| 1 | Research done | Problem framed, research questions answered | TBD | ☐ |
| 2 | Datasets selected | Shortlist evaluated, final choice made | TBD | ☐ |
| 3 | Pipeline ready | Clean data available to agents | TBD | ☐ |
| 4 | First agent working | One agent and its tools working end-to-end | TBD | ☐ |
| 5 | Full system connected | All agents orchestrated together | TBD | ☐ |
| 6 | Evaluated and documented | Results measured, write-up complete | TBD | ☐ |

### 5.2 Evaluation
- **Success metrics:** TBD (e.g. accuracy, task completion rate, latency, cost)
- **Test cases / benchmark set:** TBD
- **Baseline to compare against:** TBD
- **How we check agent outputs:** TBD (manual review, automated checks, LLM-as-judge)

### 5.3 Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| No suitable dataset found | High | Widen search, combine sources, or generate synthetic data |
| Data licence restrictions | Medium | Check licences early in phase 2 |
| Agents produce unreliable output | High | Add validation steps and evaluation tests |
| TBD | | |

---

## 6. Open Questions / Decisions Log

| Date | Question / Decision | Outcome |
|------|---------------------|---------|
| TBD  |                     |         |
