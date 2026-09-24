# Deployment Guide for DocuMind RAG

This document outlines the step-by-step procedures to deploy **DocuMind RAG** to the cloud.

---

## 🌟 1. Streamlit Community Cloud (Recommended — 100% Free & 1-Click)

Streamlit Community Cloud is the official, purpose-built hosting environment for Streamlit applications. It natively supports persistent WebSockets, Python dependencies (ChromaDB, PyTorch, HuggingFace embeddings), and automatic deployments directly from GitHub.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Enhance UI aesthetics and add Playwright test suite"
   git push origin main
   ```
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `abishxavier/rag-document-qa`.
5. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
6. Click **"Advanced settings"** and add your Secrets:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```
7. Click **"Deploy!"**.
   - Your app will be live with an SSL URL (e.g., `https://rag-document-qa.streamlit.app`).

---

## 🚀 2. Deploying to Vercel

### Architecture Note:
Vercel is primarily designed for static frontends (Next.js, React) and short-lived, stateless Serverless Functions (with a 250MB package limit and 10–60s timeouts). Streamlit, however, requires a continuous, long-running Python process and stateful WebSockets.

If you deploy to Vercel:
1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Log in to Vercel:
   ```bash
   vercel login
   ```
3. Link and deploy the project:
   ```bash
   vercel
   ```
4. Set the Environment Variable in your Vercel Dashboard or CLI:
   ```bash
   vercel env add GROQ_API_KEY
   ```
5. Deploy to production:
   ```bash
   vercel --prod
   ```

> [!NOTE]
> If Vercel displays build timeout or bundle size limits due to PyTorch and sentence-transformers, the recommended path for Vercel is to deploy a Next.js / Vite web frontend on Vercel that interacts with an external Python API or serverless Groq SDK.

---

## 🐳 3. Alternative 1-Click Platforms (Render / Hugging Face Spaces)

- **Render**: Create a "Web Service", connect the GitHub repo, set Build Command to `pip install -r requirements.txt`, and Start Command to `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Hugging Face Spaces**: Select "Streamlit Space", push your code, and define `GROQ_API_KEY` under Settings -> Variables and secrets.
