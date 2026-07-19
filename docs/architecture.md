# System Architecture Design

This document details the modular system design, agent interactions, and orchestration workflow of the **Multi-Agent AI Data Analyst**.

---

## 1. High-Level Architectural Modules

The platform is designed around four key decoupled layers:
1.  **User Interface Layer (Streamlit)**: Serves as the dashboard where users upload datasets, configure targets, trigger analysis runs, and review visual reports.
2.  **Orchestration Layer (LangGraph)**: Compiles the directed acyclic graph defining node transitions, routing states dynamically based on dataset characteristics, and writing execution checkpoints.
3.  **Agent Intelligence Layer (Specialized Agents)**: Modules coordinating specific actions (profiling, cleaning, model training, visualization, insights, reporting).
4.  **Database & Storage Layer (SQLAlchemy ORM + SQLite)**: Records datasets, logs workflow execution history, and manages checkpoint records.

---

## 2. Agent Interaction & Graph Workflow

The execution graph is structured as a **LangGraph StateGraph** consisting of nodes representing each agent's execution. Transitions between nodes are governed by conditional routes.

```mermaid
graph TD
    %% Node Definitions
    START([Start]) --> LoadNode[1. Load Dataset]
    LoadNode --> ProfileNode[2. Data Intelligence Profiler]
    ProfileNode --> RouteCondition{Routing Router}
    
    %% Conditional Routing
    RouteCondition -- "Valid Dataset" --> FeatNode[3. Feature Engineering]
    RouteCondition -- "Unsupported / Severe Errors" --> FinalNode[8. Finalize State]
    
    FeatNode --> MLNode[4. Hyperparameter tuning & ML Model training]
    MLNode --> VizNode[5. Visualization suite generation]
    VizNode --> InsightNode[6. GenAI Business Insights]
    InsightNode --> ReportNode[7. Report Generation & Compilation]
    ReportNode --> FinalNode
    FinalNode --> END([End])

    %% Checkpoint Database connections
    subgraph Persistence Layer
        DB[(SQLite Checkpoints DB)]
    end
    LoadNode <--> DB
    FeatNode <--> DB
    MLNode <--> DB
    ReportNode <--> DB
```

---

## 3. Specialized AI Agent Subsystems

Each agent operates as an isolated component, consuming state parameters and returning state updates:

```mermaid
classDiagram
    class WorkflowGraph {
        +config: WorkflowConfig
        +db: Session
        +run(dataset_path, target_column, workflow_id) WorkflowResult
    }
    class DataIntelligenceAgent {
        +run(filepath, target) DataIntelligenceResult
    }
    class FeatureEngineeringAgent {
        +run(dataframe, target) FeatureEngineeringResult
    }
    class MachineLearningAgent {
        +run(train, target, valid, valid_target) MachineLearningResult
    }
    class VisualizationAgent {
        +run(profile, feature, ml, target) VisualizationResult
    }
    class BusinessInsightAgent {
        +run(profile, feature, ml, viz) BusinessInsightResult
    }
    class ReportGenerationAgent {
        +run(profile, feature, ml, viz, business, template) ReportResult
    }

    WorkflowGraph --> DataIntelligenceAgent
    WorkflowGraph --> FeatureEngineeringAgent
    WorkflowGraph --> MachineLearningAgent
    WorkflowGraph --> VisualizationAgent
    WorkflowGraph --> BusinessInsightAgent
    WorkflowGraph --> ReportGenerationAgent
```

---

## 4. LLM Caching Strategy

To minimize latency and LLM token usage, the platform features a file-system cached retrieval layer. If the cache directory is configured, repeat requests with matching prompt signatures bypass API calls.

```mermaid
sequenceDiagram
    participant Agent as Agent / Prompter
    participant Cache as Cache Store (FileSystem)
    participant API as LLM Gateway (OpenAI/Gemini/Anthropic)

    Agent->>Cache: Check for prompt hash
    alt Cache Hit
        Cache-->>Agent: Return cached completion payload
    else Cache Miss
        Agent->>API: Execute chat request
        API-->>Agent: Return response completion
        Agent->>Cache: Save prompt hash & completion text
    end
```

---

## 5. Report Compilation Pipeline

The report generation agent builds multi-format files in three sequential phases:

```mermaid
graph LR
    A[Collect Agent Metadata] --> B[Render HTML Template via Jinja2]
    B --> C1[Compile PDF via WeasyPrint]
    B --> C2[Build DOCX via python-docx]
    B --> C3[Save HTML Output]
```
