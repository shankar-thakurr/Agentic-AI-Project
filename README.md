# 🤖 Agentic AI Project

A collection of two production-ready AI agent applications built with **LangChain**, **LangGraph**, **Groq LLMs**, and **Streamlit**.

---

## 📁 Project Structure

```
Agentic-AI-Project/
├── shopping_agent/          # 🛒 AI Shopping Assistant
│   ├── app.py               # Streamlit UI
│   ├── shopping_agent.py    # Agent logic + tools
│   ├── reviews_api.py       # Product rating functions
│   ├── setup_db.py          # SQLite DB seeder
│   ├── store.db             # SQLite product database
│   └── resources/           # Product images
│
├── telecom_chatbot/         # 📡 Telecom RAG Chatbot
│   ├── app.py               # Streamlit UI
│   ├── rag_chain.py         # RAG chain builder
│   ├── retriever.py         # Chroma multi-retriever
│   ├── ingest_faq.py        # Ingest FAQ CSV → Chroma
│   ├── ingest_pdf.py        # Ingest PDF guide → Chroma
│   ├── ingest_tickets.py    # Ingest support tickets → Chroma
│   ├── chroma_store/        # Persisted vector database
│   └── data/                # faq.csv, telecom_guide.pdf, tickets.db
│
├── pyproject.toml           # Project dependencies
├── .env                     # API keys (NOT committed)
└── .gitignore
```

---

## ⚙️ Prerequisites

- Python 3.12+
- A [Groq API Key](https://console.groq.com) (free tier available)
- `uv` package manager (recommended) or `pip`

---

## 🚀 Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/shankar-thakurr/Agentic-AI-Project.git
cd Agentic-AI-Project
```

### Step 2 — Create and activate a virtual environment

```bash
# Using uv (recommended)
uv venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### Step 3 — Install dependencies

```bash
uv pip install -r pyproject.toml
# or
pip install -e .
```

### Step 4 — Set up your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already excluded via `.gitignore`.

---

---

# 🛒 Project 1 — AI Shopping Assistant

An intelligent shopping agent that can search products, check ratings, handle image-based search, and place orders — all through a conversational Streamlit UI.

## How It Works — Step by Step

### Step 1 — Database Setup (`setup_db.py`)

Before running the app, the SQLite database must exist. The `setup_db.py` script creates three tables and seeds them with data.

```python
# setup_db.py

def create_database():
    conn = sqlite3.connect(DB_PATH)   # Creates store.db if it doesn't exist
    cursor = conn.cursor()

    # Table 1: Products (id, name, category, price, description, is_organic)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            description TEXT,
            is_organic INTEGER DEFAULT 0   # 1 = organic, 0 = non-organic
        )
    """)

    # Table 2: Reviews (linked to products via foreign key)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            rating REAL,
            reviewer_name TEXT,
            review_text TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Table 3: Orders (placed when user confirms a purchase)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            ordered_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # 32 sample products across 8 categories are inserted
    # 100+ customer reviews are seeded for realistic ratings
```

**Run it once** to create the database:

```bash
cd shopping_agent
python setup_db.py
```

---

### Step 2 — Reviews API (`reviews_api.py`)

A helper module that queries the `reviews` table and returns aggregated rating data. It is used as a tool by the AI agent.

```python
# reviews_api.py

def get_product_rating(product_id: int) -> dict:
    """Return average rating and review count for a single product."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # SQL AVG() aggregates all ratings for the product
    cursor.execute(
        "SELECT AVG(rating), COUNT(*) FROM reviews WHERE product_id = ?",
        (product_id,),
    )
    row = cursor.fetchone()
    conn.close()

    avg = round(row[0], 2) if row and row[0] is not None else 0.0
    count = row[1] if row else 0
    return {"product_id": product_id, "average_rating": avg, "review_count": count}
```

---

### Step 3 — Agent Tools (`shopping_agent.py`)

The agent is equipped with **4 tools** that it can call autonomously based on the user's request.

#### Tool 1: `search_products`

Searches the product database by keyword, with optional price and organic filters.

```python
@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Searches by keyword in name, description, and category columns.
    Optionally filters by max price and organic status.
    """
    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"           # SQL LIKE pattern for partial matching
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"      # Filter by maximum price
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"  # 1 = organic, 0 = non-organic
        params.append(1 if is_organic else 0)

    # Returns a JSON string of matching products
    return json.dumps(products)
```

#### Tool 2: `get_rating`

Fetches the average star rating for a specific product.

```python
@tool
def get_rating(product_id: int) -> str:
    """Delegates to reviews_api and returns JSON with average_rating and review_count."""
    result = get_product_rating(product_id)
    return json.dumps(result)
```

#### Tool 3: `checkout`

Places a confirmed order and saves it to the `orders` table.

```python
@tool
def checkout(product_id: int) -> str:
    """Saves an order to the database and confirms to the user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid   # Auto-generated order ID
    conn.commit()
    return f"Order #{order_id} confirmed! '{name}' ordered for ${price:.2f}."
```

#### Tool 4: `describe_product_image`

Uses a **vision LLM** (Llama 4 Scout) to analyze an uploaded product image and extract its attributes.

```python
@tool
def describe_product_image(image_path: str) -> str:
    """Encodes an image as base64 and sends it to a multimodal LLM."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()  # Base64 encode the image

    # Build a multimodal message with image + text instruction
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
        {"type": "text", "text": "Extract product_type, search_query, is_organic, description as JSON"},
    ])

    response = vision_llm.invoke([message])   # Llama 4 Scout processes the image
    return response.content                   # Returns JSON attributes
```

---

### Step 4 — Creating the Agent (`shopping_agent.py`)

LangGraph's `create_react_agent` wires the LLM and tools together into a **ReAct loop** (Reason → Act → Observe → Repeat).

```python
# Two LLMs are used:
llm        = ChatGroq(model="qwen/qwen3-32b", temperature=0)         # Main reasoning LLM
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)  # Vision LLM

agent = create_react_agent(
    model=llm,
    tools=[search_products, get_rating, checkout, describe_product_image],
    prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        # The system prompt defines 3 workflows:
        # 1. IMAGE SEARCH — analyze image → search products
        # 2. BROWSING     — search → rate → present list
        # 3. ORDERING     — confirm → checkout
        "Never place an order unless the user explicitly confirms."
    ),
)
```

**Agent flow for a typical user query:**

```
User: "I want organic honey under $15 with 4+ rating"
  └─► Agent calls search_products(query="honey", max_price=15, is_organic=True)
        └─► Agent calls get_rating(product_id=1), get_rating(product_id=7), ...
              └─► Agent filters products with rating >= 4.0
                    └─► Agent presents numbered list to user
User: "Order the first one"
  └─► Agent calls checkout(product_id=1)
        └─► Returns Order confirmation
```

---

### Step 5 — Streamlit UI (`app.py`)

The UI provides a chat interface with an image upload sidebar.

```python
# app.py

# --- Image Upload (Sidebar) ---
uploaded_file = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])

if uploaded_file and st.button("Find similar products"):
    # Save uploaded file to a temp path so the agent can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        image_path = tmp.name

    # Inject image path into the chat as a user message
    prompt = f"I uploaded a product image. Image path: {image_path}"
    st.session_state.messages.append({"role": "user", "content": prompt})

# --- Chat Input ---
if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
    # Pass full conversation history to agent for multi-turn context
    result = agent.invoke({"messages": st.session_state.messages})
    response = result["messages"][-1].content
```

### ▶️ Run the Shopping Agent

```bash
cd shopping_agent
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

---

# 📡 Project 2 — Telecom RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** customer support chatbot that answers telecom questions by retrieving relevant context from three knowledge sources: FAQs, support tickets, and a PDF guide.

## How It Works — Step by Step

### Step 1 — Ingesting FAQs (`ingest_faq.py`)

Reads `data/faq.csv` and stores each Q&A pair as a vector in ChromaDB.

```python
# ingest_faq.py

def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        # Combine question + answer into one searchable text block
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])},
        ))
    return docs

def main():
    docs = load_faq_documents(CSV_PATH)

    # Use a lightweight local embedding model (no API cost)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Clear old vectors to prevent duplicates on re-run
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection("faq")

    # Embed documents and persist to disk
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="faq",
        persist_directory=CHROMA_DIR,
    )
```

Run once:
```bash
cd telecom_chatbot
python ingest_faq.py
```

---

### Step 2 — Ingesting the PDF Guide (`ingest_pdf.py`)

Loads the telecom PDF guide, **splits it into overlapping chunks**, embeds, and stores in the `guides` collection.

```python
# ingest_pdf.py

# Load PDF — each page becomes a Document
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# Split into smaller chunks for better retrieval precision
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # Each chunk has at most 600 characters
    chunk_overlap=100,    # 100-char overlap prevents cutting context at boundaries
    separators=["\n\n", "\n", ".", " "],   # Try to split at natural paragraph/sentence breaks
)
chunks = splitter.split_documents(pages)

# Tag each chunk with its source
for i, chunk in enumerate(chunks):
    chunk.metadata["source"] = "guide"
    chunk.metadata["chunk_index"] = i

# Embed and store in 'guides' collection
Chroma.from_documents(documents=chunks, embedding=embeddings,
                      collection_name="guides", persist_directory=CHROMA_DIR)
```

Run once:
```bash
python ingest_pdf.py
```

---

### Step 3 — Ingesting Support Tickets (`ingest_tickets.py`)

Reads resolved tickets from `data/tickets.db` SQLite and stores them as vectors.

```python
# ingest_tickets.py

def load_ticket_documents(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    # Only fetch RESOLVED tickets — these have proven solutions
    rows = conn.execute("SELECT * FROM tickets WHERE status = 'resolved'").fetchall()

    docs = []
    for row in rows:
        # Combine issue + resolution for full context retrieval
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": "ticket", "ticket_id": row["ticket_id"]},
        ))
    return docs
```

Run once:
```bash
python ingest_tickets.py
```

---

### Step 4 — Building the Retriever (`retriever.py`)

Creates a **merged retriever** that queries all three Chroma collections simultaneously.

```python
# retriever.py

def build_retriever(k_faq=3, k_tickets=3, k_guides=3) -> RunnableLambda:
    # All three collections use the same embedding model for consistent vector space
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Connect to each collection
    faq_store     = Chroma(collection_name="faq",     embedding_function=embeddings, persist_directory=CHROMA_DIR)
    tickets_store = Chroma(collection_name="tickets", embedding_function=embeddings, persist_directory=CHROMA_DIR)
    guides_store  = Chroma(collection_name="guides",  embedding_function=embeddings, persist_directory=CHROMA_DIR)

    # Each retriever fetches the top-k most similar documents
    faq_retriever     = faq_store.as_retriever(search_kwargs={"k": k_faq})       # top 3 FAQs
    tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets}) # top 3 tickets
    guides_retriever  = guides_store.as_retriever(search_kwargs={"k": k_guides})  # top 3 guide chunks

    def retrieve(query: str) -> list[Document]:
        # Merge results from all three sources into one list (max 9 docs)
        return (
            faq_retriever.invoke(query)
            + tickets_retriever.invoke(query)
            + guides_retriever.invoke(query)
        )

    return RunnableLambda(retrieve)   # Wrap as a LangChain Runnable
```

---

### Step 5 — Building the RAG Chain (`rag_chain.py`)

Wires the retriever, prompt, LLM, and output parser into a single pipeline using LangChain's `|` (pipe) operator.

```python
# rag_chain.py

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Use ONLY the context below to answer the customer's question.
The context comes from:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information, say so clearly
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""

def _format_docs(docs: list[Document]) -> str:
    """Formats retrieved docs by labelling each with its source [FAQ], [TICKET], [GUIDE]."""
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),  # System context + retrieved docs
        ("human", "{question}"),    # User's actual question
    ])

    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_tokens=1024)

    # The RAG pipeline:
    # 1. User question → retriever fetches relevant docs
    # 2. Docs are formatted and injected into the system prompt as {context}
    # 3. Prompt is sent to LLM
    # 4. LLM response is parsed to plain string
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

**Data flow diagram:**

```
User Question
     │
     ▼
 Retriever ──────────────────────────────────────────────────┐
   ├── faq_retriever     → top 3 FAQ Q&A pairs               │
   ├── tickets_retriever → top 3 resolved support tickets     │
   └── guides_retriever  → top 3 PDF guide chunks             │
                                                              ▼
                                                    Format as [SOURCE]\ncontent
                                                              │
                                                              ▼
                                              Inject into System Prompt {context}
                                                              │
                                                              ▼
                                                   Qwen3-32B on Groq
                                                              │
                                                              ▼
                                                     Answer to User
```

---

### Step 6 — Streamlit UI (`app.py`)

Provides a clean chat interface with sample question shortcuts in the sidebar.

```python
# app.py

# Cache the RAG chain so it is only built once (embedding model loads once)
@st.cache_resource
def get_chain():
    return build_chain()

# Sidebar sample questions — clicking one instantly sends it as a message
for q in SAMPLE_QUESTIONS:
    if st.button(q, key=f"sample_{q[:20]}"):
        st.session_state.pending_question = q   # Store for processing on next rerun

# Stream the response token by token for a better UX
if question:
    chain = get_chain()
    response = st.write_stream(chain.stream(question))   # Streams response in real time
```

### ▶️ Run the Telecom Chatbot

```bash
cd telecom_chatbot
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

---

## 🔑 Environment Variables

| Variable | Description | Used In |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) | Both projects |

Create a `.env` file in the root:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `langchain` | LLM orchestration framework |
| `langgraph` | ReAct agent loop for shopping agent |
| `langchain-groq` | Groq LLM integration (Qwen3-32B, Llama 4) |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-huggingface` | Local sentence embedding model |
| `sentence-transformers` | `all-MiniLM-L6-v2` embedding model |
| `chromadb` | Persistent local vector database |
| `streamlit` | Web UI framework |
| `python-dotenv` | Load `.env` API keys |
| `pypdf` | PDF loading for telecom guide |
| `pandas` | CSV loading for FAQ ingestion |
| `sqlite3` | Built-in Python SQLite for products/orders/tickets |

---

## 💡 Models Used

| Model | Provider | Used For |
|---|---|---|
| `qwen/qwen3-32b` | Groq | Main reasoning in both projects |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Groq | Image analysis in shopping agent |
| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace (local) | Text embeddings in telecom chatbot |

---

## 🛡️ Security Notes

- Never commit your `.env` file — it's excluded in `.gitignore`
- The `.env` file inside `telecom_chatbot/` should also not be committed (already ignored)
- Rotate your Groq API key if it was ever accidentally exposed

---

## 📄 License

MIT License — free to use, modify, and distribute.
