# Temperature Scaling Blocks

Detailed information about temperature scaling for flattening probability distributions.

```mermaid
flowchart LR
    A[Logits] --> B[Divide by T=1000]
    B --> C[Exponential Summation]
    C --> D[Energy Score]
```

[Back to README](../README.md)
