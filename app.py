import streamlit as st
import json
import re
import src.retriever as retriever
import src.verifier as verifier

st.set_page_config(
    page_title="News Authenticity Verifier",
    page_icon="📰",  # Site icon
    layout="centered",
)

@st.cache_resource
def load_resources():
    retriever.initialize_resources()
    verifier.initialize_resources()
    return retriever, verifier

def extract_json(text):
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()

def display_verdict(verdict_data):
    verdict = verdict_data.get("verdict", "Unverifiable").strip().capitalize()
    reasoning = verdict_data.get("reasoning", "No reasoning provided.").strip()

    # Color & emoji mapping
    verdict_styles = {
        "Real": ("✅ Real", "#2ecc71"),
        "Fake": ("❌ Fake", "#e74c3c"),
        "Unverifiable": ("⚠️ Unverifiable", "#f1c40f"),
    }

    display_text, color = verdict_styles.get(verdict, ("⚠️ Unverifiable", "#f1c40f"))

    st.markdown(
        f"""
        <div style="
            background-color:{color}20;
            border-left: 6px solid {color};
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        ">
            <h3 style="color:{color}; margin:0;">{display_text}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background-color:#f8f9fa;
            padding:1rem;
            border-radius:10px;
            margin-top:1rem;
            box-shadow:0 1px 4px rgba(0,0,0,0.1);
        ">
            <h4>🧠 Reasoning:</h4>
            <p style="font-size:1rem; line-height:1.5; color:#333;">{reasoning}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    st.title("📰 News Authenticity Verifier")
    st.markdown(
        """
        <p style="color:#6c757d; font-size:1rem;">
        Enter a news claim or statement below to verify its authenticity against factual sources.
        </p>
        """,
        unsafe_allow_html=True,
    )

    retriever, verifier = load_resources()

    claim = st.text_area("✍️ Enter a claim to verify:", height=180, placeholder="e.g. Trump is the current US president")

    if st.button("🔍 Verify Claim"):
        if claim.strip():
            with st.spinner("🔎 Retrieving relevant articles..."):
                evidences = retriever.similar_articles(claim, k=3)

            with st.spinner("🧩 Analyzing claim with AI fact-checking model..."):
                verdict = verifier.verify_claim(claim, evidences)
                print(verdict)

            try:
                verdict_data = json.loads(extract_json(verdict))
                st.success("✅ Analysis Complete")
                display_verdict(verdict_data)
            except json.JSONDecodeError:
                st.error("⚠️ Could not parse the model response as JSON.")
                st.text(verdict)
        else:
            st.warning("⚠️ Please enter a claim to verify.")
