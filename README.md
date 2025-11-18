# 30 Days Learning Generative AI

A comprehensive learning journey through modern generative AI concepts, tools, and production systems.

---

## WEEK 1 — FOUNDATIONS: LLMS, EMBEDDINGS, RAG, PROMPTING

### Day 1 — Modern LLM Landscape + Architecture
*(GPT-4.1, o1, Claude 3.5, Llama 3, Mistral)*

#### Concepts
- Transformer architecture refresh (encoder, decoder, self-attention)
- Differences between GPT, Claude, Llama, Mistral, Gemini
- Context window, latency, throughput, batching
- Why some models reason longer (DELIBERATE mode)

#### Official Docs
- [OpenAI Models Overview](https://platform.openai.com/docs/models)
- [Anthropic Claude Model Guide](https://docs.anthropic.com/en/docs/about-claude)
- [Meta Llama 3 Tech Overview](https://llama.meta.com/)

#### Research Papers
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- ["GPT-4 Technical Report"](https://cdn.openai.com/papers/gpt-4.pdf)

#### YouTube
- [Andrej Karpathy — "The State of GPT"](https://www.youtube.com/watch?v=bZQun8Y4L2A)
- [OpenAI Dev Day (must-watch)](https://www.youtube.com/@OpenAI)

#### Reddit
- r/MachineLearning "LLM Models Comparison"
- r/LocalLLaMA "Llama 3 deep dive threads"

---

### DAY 2 — Tokenization, Embeddings & Vector Math

#### Concepts
- BPE tokenization
- Embedding spaces
- Cosine similarity vs dot-product
- Dimensionality selection
- Embedding drift + why embeddings change

#### Official Docs
- [OpenAI Embeddings Overview](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)

#### Research Papers
- ["Sentence-BERT: Sentence Embeddings using Siamese BERT Networks"](https://arxiv.org/abs/1908.10084)

#### Medium
- "How Embeddings Really Work"
- "Choosing Embedding Models for RAG"

#### YouTube
- [Pinecone: Intro to Vector Databases](https://www.youtube.com/@PineconeIO)

#### Reddit
- r/VectorDB "Best embeddings 2024–25"

---

### Day 3 — Prompt Engineering (Production Grade)

#### Concepts
- System prompt vs user prompt vs assistant prompt
- Few-shot prompting
- Style + persona prompts
- When to use chain-of-thought
- How to reduce hallucinations

#### Official Docs
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)

#### Research Papers
- ["Large Language Models are Zero-Shot Reasoners"](https://arxiv.org/abs/2205.11916)

#### YouTube
- [Prompt Engineering Explained](https://www.youtube.com/@deep_learning_ai)

#### Reddit
- r/PromptEngineering — best prompts discussion

---

### Day 4 — RAG Basics (Chunking, Indexing, Retrieval)

#### Concepts
- Types of chunking
- How retrieval works
- Top-k retrieval
- Hybrid search (BM25 + embeddings)
- Retrieval bottlenecks

#### Official Docs
- [OpenAI Retrieval Guide](https://platform.openai.com/docs/guides/retrieval)
- [Weaviate RAG 101](https://weaviate.io/developers/weaviate)

#### Research Papers
- ["RAG: Retrieval Augmented Generation"](https://arxiv.org/abs/2005.11401)

#### Medium
- "What is RAG? Full Breakdown"
- "Why chunking matters"

#### YouTube
- Weaviate Hybrid Search Tutorial

---

### Day 5 — Context Optimization & Chunking Theory

#### Concepts
- Chunking strategies: recursive, semantic, sliding window
- Overlap size tuning
- Context window usage
- Prompt compression

#### Official Docs
- [LlamaIndex Chunking Guide](https://docs.llamaindex.ai/en/latest/module_guides/loading/chunking/)

#### Research Papers
- ["Lost in the Middle" (context dilution)](https://arxiv.org/abs/2307.03172)

#### YouTube
- LlamaIndex Deep Dive on Chunking

#### Reddit
- r/RAG — chunking best practices threads

---

### Day 6 — Vector Databases Deep Dive

#### Concepts
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- Scalar quantization
- Metadata indexing
- Vector sharding and replication
- Cost and latency optimization

#### Official Docs
- [Pinecone Architecture](https://docs.pinecone.io/docs)
- [Milvus In-Depth](https://milvus.io/)

#### Research Papers
- [HNSW Paper](https://arxiv.org/abs/1603.09320)

#### Reddit
- r/DataEngineering "Vector DB comparisons"

---

### Day 7 — Week 1 Consolidation ✅

#### Review Topics
- ✓ LLM basics & architecture
- ✓ Embeddings & vector math
- ✓ RAG fundamentals
- ✓ Prompt engineering techniques
- ✓ Chunking strategies
- ✓ Hybrid search methods

#### Watch & Study
- Pinecone RAG Playlist
- Weaviate RAG Playlist

---

## WEEK 2 — ADVANCED RAG, RERANKING, EVALS, GUARDRAILS

**Focus:** Production-grade 2025 RAG systems

### Day 8 — Reranking (Cohere, Voyage AI, OpenAI)

#### Concepts
- Cross-encoder rerankers
- Precision improvements over BM25
- Cost vs latency trade-offs

#### Official Docs
- [Cohere ReRank Guide](https://docs.cohere.com/docs/rerank)

#### Research Papers
- ["ColBERT: Efficient Passage Retrieval"](https://arxiv.org/abs/2004.12832)

---

### Day 9 — Query Transformation (HyDE, Query Rewriting)

#### Concepts
- HyDE (Hypothetical Document Embeddings)
- Multi-turn query compression
- Query expansion & rewriting

#### Research Papers
- ["HyDE: Hypothetical Document Embeddings"](https://arxiv.org/abs/2212.10496)

#### YouTube
- RAG Enhancements (OpenAI Dev Sessions)

---

### Day 10 — Advanced RAG Architectures

#### Concepts
- Multi-vector indexes
- Routing & dynamic RAG
- Agentic RAG patterns
- Multi-stage retrieval pipelines

#### Official Docs
- [LlamaIndex Advanced RAG](https://docs.llamaindex.ai/en/stable/understanding/)

#### Research Papers
- ["RAG Triad: Retrieval, Reranking, Generation"](https://arxiv.org/pdf/2312.06646)

---

### Day 11 — RAG Evaluation (Metrics & Tools)

#### Concepts
- Answer correctness metrics
- Hallucination rate detection
- Retrieval hit rate analysis
- Ground truth creation strategies

#### Official Docs
- [OpenAI Evals Framework](https://platform.openai.com/docs/guides/evals)

#### Tools
- LangSmith RAG Evaluations
- TrueLens

#### Research Papers
- ["RAGAS Benchmark for RAG"](https://arxiv.org/abs/2403.03390)

---

### Day 12 — Guardrails & Safety

#### Concepts
- AWS Bedrock Guardrails
- Content moderation
- Output validation & filtering
- Safety classifiers

#### Official Docs
- [Anthropic Safety Guidelines](https://docs.anthropic.com/en/docs/safety)
- [Guardrails AI Framework](https://docs.guardrailsai.com/)

#### Reddit
- r/LLM — safety discussions & best practices

---

### Day 13 — Memory Systems (Short-term, Long-term, Associative)

#### Concepts
- Memory buffers & context windows
- Episodic memory (conversation history)
- Entity memory (relationship tracking)
- Long-term vector memory (semantic recall)

#### Official Docs
- [LangChain Memory Module](https://python.langchain.com/docs/modules/memory)

#### Research Papers
- ["MemGPT: Towards LLMs as Operating Systems"](https://arxiv.org/abs/2310.08560)

---

### Day 14 — Week 2 Consolidation ✅

#### Study Resources
- LangChain RAG Series (YouTube)
- OpenAI RAG Deep Dives
- Cohere Reranker Series

---

## WEEK 3 — AGENTS, TOOLS, PLANNERS, MULTI-AGENTS

**Focus:** Agent frameworks & cognitive architectures

### Day 15 — Agents & Tool Calling (OpenAI, Anthropic)

#### Concepts
- Tool schema definition
- Multi-tool agent orchestration
- Agent loop mechanics
- Planner + executor architecture

#### Official Docs
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Claude Tool Use API](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

#### Research Papers
- ["Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)

---

### Day 16 — ReAct, ToT, CoT & Tree Search

#### Concepts
- ReAct Framework (Reason + Act)
- CoT (Chain of Thought) prompting
- Tree-of-Thoughts planning
- Search tree exploration

#### Research Papers
- ["ReAct: Synergizing Reasoning and Acting"](https://arxiv.org/abs/2210.03629)
- ["Tree of Thoughts: Deliberate Problem Solving"](https://arxiv.org/abs/2305.10601)

---

### Day 17 — Multi-Agent Systems (LangGraph)

#### Concepts
- Agent graph topology
- State propagation & transitions
- Event loop execution
- Error handling & recovery

#### Official Docs
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph/)

#### Research Papers
- ["AutoGen: Enabling Next-Gen LLM Applications"](https://arxiv.org/abs/2308.08155)

---

### Day 18 — Semantic Kernel (Enterprise Agents)

#### Concepts
- Planners (action sequencing)
- Skills (reusable functions)
- Memories (context management)
- Pipelines (orchestration)

#### Official Docs
- [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)

#### YouTube
- Semantic Kernel Deep Dive Series by Microsoft

---

### Day 19 — LlamaIndex Agents

#### Concepts
- Toolsets & agent tools
- Executors & execution strategies
- Distributed agent patterns

#### Official Docs
- [LlamaIndex Agent Examples](https://docs.llamaindex.ai/en/latest/examples/agent/)

---

### Day 20 — Agent Safety, Misuse & Validation

#### Concepts
- Escape prevention mechanisms
- Objective misalignment detection
- Tool abuse prevention
- Validation strategies

#### Research Papers
- ["AgentBench: Evaluating LLMs as Agents"](https://arxiv.org/abs/2308.03688)

---

### Day 21 — Week 3 Consolidation ✅

#### Recommended Learning
- Microsoft Semantic Kernel Playlist
- Anthropic Agent Architecture Demos
- LangGraph Conference Talks

---

## WEEK 4 — LLM SYSTEM DESIGN, DEPLOYMENT, COST, EVALS

**Focus:** Infrastructure, MLOps, observability & production systems

### Day 22 — LLM API Architectures (Production)

#### Concepts
- Multi-model routing & selection
- Load balancing strategies
- Latency optimization
- Global scaling patterns

#### Official Docs
- [OpenAI Best Practices](https://platform.openai.com/docs/guides)
- [AWS Bedrock Architecture](https://aws.amazon.com/bedrock/)

---

### Day 23 — LLMOps & Observability

#### Concepts
- Distributed tracing
- Prompt versioning & management
- Log aggregation & analysis
- Evaluation dashboards

#### Tools
- LangSmith (debugging & optimization)
- Weights & Biases LLMOps
- Arize AI Phoenix

---

### Day 24 — Deployment on AWS (Serverless + Containers)

#### Concepts
- Lambda vs EC2 vs EKS trade-offs
- Auto-scaling configurations
- API Gateway patterns
- IAM permissions & quotas

#### Official Docs
- [AWS Lambda Best Practices](https://aws.amazon.com/lambda/)
- [AWS RDS Vector Extensions](https://aws.amazon.com/rds/)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/s3/)

---

### Day 25 — Cost Optimization

#### Concepts
- Token count optimization
- Adaptive model selection
- Generation limits
- Caching layers (prompt, KV)

#### Official Docs
- [OpenAI Cost Optimization Guide](https://platform.openai.com/docs/guides/best-practices/cost)

#### Community
- r/devops "LLM cost breakdowns & strategies"

---

### Day 26 — Fine-tuning Theory

#### Concepts
- Fine-tuning vs prompt engineering trade-offs
- SFT (Supervised Fine-Tuning) datasets
- RLHF (Reinforcement Learning from Human Feedback)
- LoRA (Low-Rank Adaptation) for efficiency

#### Research Papers
- ["LoRA: Low-Rank Adaptation for Large Language Models"](https://arxiv.org/abs/2106.09685)

---

### Day 27 — Evaluation Theory (Deep Dive)

#### Concepts
- Human-in-the-loop evaluation
- Synthetic evaluation methods
- Task-specific benchmarks
- Policy testing frameworks

#### Research & Benchmarks
- [HELM Benchmark](https://crfm.stanford.edu/helm/latest/)

---

### Day 28 — Scaling RAG & Agents in Production

#### Concepts
- Multi-tenant RAG architecture
- Distributed vector search
- High-throughput LLM routing
- Failure recovery & redundancy

#### YouTube
- Weaviate Enterprise RAG Webinars
- Pinecone Scaling Deep Dive

---

### Day 29 — AI Safety, Governance & Compliance

#### Concepts
- PII (Personally Identifiable Information) protection
- Data governance frameworks
- GDPR, HIPAA, SOC 2 compliance
- Model card documentation

#### Resources
- EU AI Act Summary
- OpenAI Safety Guidelines

---

### Day 30 — Final Revision & Mastery ✅

#### Master Reference Playlists
- [OpenAI Developer Sessions](https://www.youtube.com/@OpenAI)
- [LangChain + LangGraph](https://www.youtube.com/@LangChain)
- [Microsoft Semantic Kernel](https://www.youtube.com/c/Microsoft)
- [Weaviate RAG Series](https://www.youtube.com/@Weaviate)

#### Must-Read Foundational Papers
1. [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
2. [Toolformer: Teaching LLMs to Use Tools](https://arxiv.org/abs/2302.04761)
3. [RAG Triad: Retrieval, Reranking, Generation](https://arxiv.org/pdf/2312.06646)
4. [Lost in the Middle: Context Window Analysis](https://arxiv.org/abs/2307.03172)
5. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
6. [Tree of Thoughts: Deliberate Problem Solving](https://arxiv.org/abs/2305.10601)
7. [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)

---

## 🎯 Learning Path Summary

| Week | Focus | Key Skills |
|------|-------|-----------|
| **Week 1** | Foundations | LLMs, Embeddings, RAG, Prompting |
| **Week 2** | Advanced RAG | Reranking, Evaluation, Safety |
| **Week 3** | Agents | Tool Calling, Multi-Agents, Planning |
| **Week 4** | Production | Deployment, Cost Optimization, Scaling |

---

## 📚 Additional Resources

### Official Documentation
- OpenAI API Documentation
- Anthropic Claude API
- LangChain Framework
- LlamaIndex

### Communities
- r/MachineLearning
- r/LocalLLaMA
- r/PromptEngineering
- r/DataEngineering

---

**Happy Learning! 🚀**

*Last Updated: November 2025*
