import os

# Create assets folder
os.makedirs("assets", exist_ok=True)

# Generate SVG banner
svg_content = """<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:rgb(255,100,100);stop-opacity:1" />
      <stop offset="100%" style="stop-color:rgb(100,100,255);stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grad1)">
    <animate attributeName="opacity" values="0.8;1;0.8" dur="3s" repeatCount="indefinite" />
  </rect>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="40" font-family="Arial" fill="white" font-weight="bold">
    Awesome OOD Detection
    <animate attributeName="y" values="45%;55%;45%" dur="5s" repeatCount="indefinite" />
  </text>
  <text x="50%" y="70%" dominant-baseline="middle" text-anchor="middle" font-size="20" font-family="Arial" fill="white">
    Safety & Reliability in AI
  </text>
</svg>"""

with open("assets/banner.svg", "w") as f:
    f.write(svg_content)

pages = [
    {
        "filename": "msp_baseline.md",
        "title": "The Baseline Softmax Probability Era",
        "content": "Detailed information about Maximum Softmax Probability (MSP).",
        "diagram": "flowchart TD\n    A[Input] --> B[Neural Network]\n    B --> C[Softmax Layer]\n    C --> D{Max Prob > Threshold?}\n    D -- Yes --> E[In-Distribution]\n    D -- No --> F[OOD]"
    },
    {
        "filename": "post_hoc_logit.md",
        "title": "The Post-Hoc Logit & Energy Scaling Era",
        "content": "Detailed information about Energy-Based OOD Detection.",
        "diagram": "flowchart TD\n    A[Input] --> B[Logits]\n    B --> C[Temperature Scaling]\n    C --> D[Energy Calculation]\n    D --> E{Energy < Threshold?}\n    E -- Yes --> F[In-Distribution]\n    E -- No --> G[OOD]"
    },
    {
        "filename": "generative_density.md",
        "title": "The Generative Density Estimation Era",
        "content": "Detailed information about using generative models for density estimation.",
        "diagram": "flowchart TD\n    A[Input] --> B[PixelCNN/Normalizing Flow]\n    B --> C[Likelihood Score]\n    C --> D[Likelihood Ratio Correction]\n    D --> E[OOD Classification]"
    },
    {
        "filename": "monosemantic_vlm.md",
        "title": "The Monosemantic Feature Dictionary & VLM Era",
        "content": "Detailed information about Sparse Autoencoders and Concept Auditing.",
        "diagram": "flowchart TD\n    A[Input Tokens] --> B[LLM Latent State]\n    B --> C[Sparse Autoencoder]\n    C --> D[Monosemantic Features]\n    D --> E[Anomaly Detection]"
    },
    {
        "filename": "post_hoc_score_discrimination.md",
        "title": "Post-Hoc Score Discrimination",
        "content": "Detailed information about Model-Free Refitting.",
        "diagram": "flowchart LR\n    A[Frozen Model] --> B[Extract Logits]\n    B --> C[Compute Mahalanobis/Energy]"
    },
    {
        "filename": "outlier_exposure.md",
        "title": "Outlier Exposure",
        "content": "Detailed information about Adversarial Data-Centric OOD.",
        "diagram": "flowchart LR\n    A[In-Distribution Data] --> B[Train Loop]\n    C[Outlier Dataset] --> B\n    B --> D[Penalize High Confidence on Outliers]"
    },
    {
        "filename": "self_supervised_multimodal.md",
        "title": "Self-Supervised Multi-Modal Anomaly Detection",
        "content": "Detailed information about utilizing CLIP/SigLIP for OOD.",
        "diagram": "flowchart LR\n    A[Image] --> B[Vision Encoder]\n    C[Text Prompts] --> D[Text Encoder]\n    B --> E[Cosine Similarity]\n    D --> E\n    E --> F[Anomaly Detection]"
    },
    {
        "filename": "test_time_compute.md",
        "title": "Test-Time Compute Reasoning Auditing",
        "content": "Detailed information about reasoning models and self-correction.",
        "diagram": "flowchart TD\n    A[Prompt] --> B[Reasoning Trace]\n    B --> C{Logical Contradiction?}\n    C -- Yes --> D[Flag as OOD]\n    C -- No --> E[Final Answer]"
    },
    {
        "filename": "temperature_scaling.md",
        "title": "Temperature Scaling Blocks",
        "content": "Detailed information about temperature scaling for flattening probability distributions.",
        "diagram": "flowchart LR\n    A[Logits] --> B[Divide by T=1000]\n    B --> C[Exponential Summation]\n    C --> D[Energy Score]"
    },
    {
        "filename": "mahalanobis_centroids.md",
        "title": "Mahalanobis Covariance Centroids",
        "content": "Detailed information about tracking high-dimensional distance.",
        "diagram": "flowchart TD\n    A[Layer Embeddings] --> B[Compute Mean Vector]\n    A --> C[Compute Covariance]\n    B & C --> D[Mahalanobis Distance]"
    },
    {
        "filename": "latency_overhead_wall.md",
        "title": "The Latency-Overhead Wall of Deep Generative Pipelines",
        "content": "Detailed information about system processing latency and mitigations.",
        "diagram": "flowchart TD\n    A[Incoming Request] --> B[Primary Model]\n    A --> C[Generative Density Network (Slow)]\n    C -.-> D[Latency Bottleneck]"
    },
    {
        "filename": "false_positive_capacity_drain.md",
        "title": "The False-Positive Capacity Drain",
        "content": "Detailed information about Refusal Over-generalization and the alignment tax.",
        "diagram": "flowchart TD\n    A[Creative Request] --> B[Conservative Threshold]\n    B --> C[False Positive Flag]\n    C --> D[Degraded User Utility]"
    },
    {
        "filename": "autonomous_driving.md",
        "title": "Autonomous Driving Perception Fleet Defensives",
        "content": "Detailed information about BEV Perception and LiDAR anomaly detection.",
        "diagram": "flowchart LR\n    A[Camera/LiDAR] --> B[Perception Stack]\n    B --> C[OOD Detection]\n    C -- Novel Hazard --> D[Defensive Braking]"
    },
    {
        "filename": "clinical_pathology.md",
        "title": "Mission-Critical Clinical Pathology Diagnostic Safeguards",
        "content": "Detailed information about regulating medical diagnostic bots.",
        "diagram": "flowchart LR\n    A[Tissue Scan] --> B[Vision Encoder]\n    B --> C[OOD Gate]\n    C -- Rare Mutation --> D[Halt Automation & Alert Clinician]"
    },
    {
        "filename": "enterprise_fraud.md",
        "title": "Enterprise Financial Fraud & Cyber-Security Intrusion Detection",
        "content": "Detailed information about monitoring cloud API transactions for outliers.",
        "diagram": "flowchart LR\n    A[API Logs] --> B[Density Scoring Module]\n    B --> C[High-Energy Outlier?]\n    C -- Yes --> D[Flag Cyber-Attack]"
    }
]

os.makedirs("details", exist_ok=True)
for page in pages:
    with open(f"details/{page['filename']}", "w") as f:
        f.write(f"# {page['title']}\n\n{page['content']}\n\n```mermaid\n{page['diagram']}\n```\n\n[Back to README](../README.md)\n")

print("Files generated successfully.")
