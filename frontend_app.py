"""
Streamlit Frontend for Regulatory Compliance Q&A Model
Demonstrates the fine-tuned Llama model with case studies
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time
import textwrap

# Page configuration
st.set_page_config(
    page_title="Regulatory Compliance Q&A Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .case-study-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2ca02c;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Case studies
CASE_STUDIES = {
    "Case Study 1: Capital Adequacy Compliance": {
        "scenario": """
        **Scenario:** Rohan (32) is a salaried software engineer in Mumbai earning ₹1,50,000/month.
        He has one small personal loan EMI of ₹10,000 and wants to buy a ₹80,00,000 apartment with
        a ₹60,00,000 home loan for 20 years. He has a clean repayment history (CIBIL 782) and complete KYC.
        Assumed interest rate: 9.0% p.a. (floating).
        He wants to know whether he would generally be eligible and under what typical conditions.

        **Question:** Is Rohan likely eligible for a ₹60,00,000 home loan with standard documentation,
        and what reasons/conditions would typically apply?
        """,
        "question": "What are the capital adequacy requirements for housing finance companies?",
        "expected_domain": "Capital Adequacy, Regulatory Compliance"
    },
    "Case Study 2: KYC Update for Minor Account": {
        "scenario": """
        **Scenario:** Aisha opened a home loan account as a minor co‑applicant with her mother at age 17.
        She has turned 18 and wants to take a small top‑up and become primary applicant on the account.
        Her address recently changed for college and she has now obtained a PAN card.
        Assumed interest rate for top‑up: 9.2% p.a.; tenure balance 20 years.

        **Question:** What KYC updates and additional checks are typically required when Aisha becomes an adult
        and seeks a role change/top‑up on the existing loan?
        """,
        "question": "What is required for KYC updates when a minor becomes an adult?",
        "expected_domain": "KYC/AML, Customer Due Diligence"
    },
    "Case Study 3: Exposure Limit Compliance": {
        "scenario": """
        **Scenario:** Vivek (40), a self‑employed trader in Ahmedabad, reports average net income of ₹1,10,000/month
        (post‑normalization). He already pays EMIs of ₹25,000. He wishes to purchase a row house worth ₹65,00,000
        with a loan of ₹55,00,000 for 20 years. CIBIL is 735; business vintage 2.5 years.
        Assumed interest rate: 9.5% p.a. (self‑employed slab).

        **Question:** Is Vivek likely eligible for the requested loan amount, and what factors (FOIR, LTV,
        credit stability) could lead to approval/decline or conditional sanction?
        """,
        "question": "What is the total exposure limit to housing finance and CRE-Residential Housing for all StCBs/DCCBs?",
        "expected_domain": "Risk Management, Exposure Limits"
    },
    "Case Study 4: Registration Cancellation": {
        "scenario": """
        **Scenario:** Sunita (36), a salaried teacher in Delhi earning ₹2,00,000/month, has a low credit score of 640
        due to past delays but no active defaults. She wants a ₹80,00,000 loan on a ₹1,20,00,000 apartment for 20 years.
        Current EMIs are ₹15,000. KYC is complete. Assumed interest rate: 9.3% p.a.

        **Question:** Is Sunita likely to be eligible? Explain how low CIBIL affects the decision and what conditions
        (higher margin, additional guarantor, seasoning) could help.
        """,
        "question": "Under what conditions can the Reserve Bank cancel a certificate of registration for a housing finance institution?",
        "expected_domain": "Registration, Regulatory Enforcement"
    },
    "Case Study 5: Securities Market Value": {
        "scenario": """
        **Scenario:** Raj and Meera (27/26), both salaried in Jaipur with combined income ₹1,10,000/month and CIBIL 765/770,
        want to buy a ₹22,00,000 apartment with a ₹19,00,000 loan for 25 years. Existing EMIs are ₹5,000.
        They want to know if they qualify comfortably and what typical documents/conditions will apply.
        Assumed interest rate: 9.1% p.a.

        **Question:** Are Raj and Meera likely eligible for the requested loan? Provide reasons (FOIR, LTV, co‑applicant
        income) and any standard conditions the lender may impose.
        """,
        "question": "What is the market value requirement for securities held for depositors?",
        "expected_domain": "Securities, Deposit Protection"
    }
}

@st.cache_resource
def load_model(model_path: str, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_adapters: bool = False):
    """Load the fine-tuned model (cached for performance)."""
    try:
        st.info("🔄 Loading model... This may take a few minutes on first load.")
        
        # Check for Hugging Face token
        import os
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            st.warning("⚠️ HUGGINGFACE_HUB_TOKEN not set. You may need to authenticate.")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Check if CUDA is available
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        # Load LoRA adapters (optional)
        if use_adapters:
            if os.path.exists(model_path):
                model = PeftModel.from_pretrained(model, model_path)
            else:
                st.warning(f"⚠️ Adapters path not found: {model_path}. Proceeding without adapters.")
        
        return model, tokenizer, use_cuda
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, False

def format_prompt(question: str, context: str = None) -> str:
    """Format a regulatory question for the model."""
    if context:
        prompt = f"""### Instruction:
You are a compliance analyst. Answer regulatory questions based on the provided context.

### Input:
Context: {context}

Question: {question}

### Response:
"""
    else:
        prompt = f"""### Instruction:
You are a compliance analyst. Answer regulatory questions based on your training.

### Input:
Question: {question}

### Response:
"""
    return prompt

def format_case_study_eligibility_prompt(scenario_markdown: str, explicit_question: str = None) -> str:
    """Prompt template to decide loan eligibility with reasons from case-study background."""
    base = textwrap.dedent(scenario_markdown).strip()
    question_line = f"\nAdditional Question: {explicit_question}\n" if explicit_question else "\n"
    return f"""### Instruction:
You are a senior mortgage underwriter for Indian lenders. Based on the background below, decide whether the case
is ELIGIBLE for a home loan. Provide a clear Yes/No decision with concise reasons referencing KYC, credit,
FOIR/affordability, LTV/margin, documentation, and any regulatory constraints relevant to the scenario.

### Background:
{base}
{question_line}
Output format:
- Eligibility: Yes/No
- Reasons: 3–6 short bullets grounded in prudent underwriting
- Conditions (if any): brief list of covenants or documents required

### Response:
"""

def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512, use_cuda: bool = False):
    """Generate an answer using the fine-tuned model."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to device
    device = "cuda" if use_cuda else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    response_start = full_response.find("### Response:\n")
    if response_start != -1:
        answer = full_response[response_start + len("### Response:\n"):].strip()
    else:
        answer = full_response[len(prompt):].strip()
    
    return answer

def estimate_emi(principal: float, annual_rate_percent: float, tenure_years: float) -> float:
    """Estimate monthly EMI using standard amortization formula."""
    if principal <= 0 or annual_rate_percent <= 0 or tenure_years <= 0:
        return 0.0
    monthly_rate = (annual_rate_percent / 100.0) / 12.0
    n = int(tenure_years * 12)
    try:
        emi = principal * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
        return float(emi)
    except Exception:
        return 0.0

def compute_foir(monthly_income: float, existing_emis: float, proposed_emi: float) -> float:
    """FOIR = (existing EMIs + proposed EMI) / net monthly income."""
    denom = max(monthly_income, 1e-6)
    return float((existing_emis + proposed_emi) / denom)

def compute_ltv(loan_amount: float, property_value: float) -> float:
    """LTV ratio = loan amount / property value."""
    denom = max(property_value, 1e-6)
    return float(loan_amount / denom)

def build_eligibility_context(data: dict) -> str:
    """Create a compact context string from applicant details."""
    fields = [
        f"Age: {data.get('age')}",
        f"Employment: {data.get('employment_type')} ({data.get('employment_stability_years')} years)",
        f"Monthly Income: ₹{data.get('monthly_income')}",
        f"Existing EMIs: ₹{data.get('existing_emis')}",
        f"CIBIL: {data.get('cibil_score')}",
        f"Property Type: {data.get('property_type')}",
        f"Property Value: ₹{data.get('property_value')}",
        f"Loan Amount: ₹{data.get('loan_amount')}",
        f"Tenure: {data.get('tenure_years')} years",
        f"KYC Complete: {data.get('kyc_complete')}",
        f"NPA/Default History: {data.get('npa_history')}",
        f"City/State: {data.get('location')}",
    ]
    return " | ".join(fields)

def format_eligibility_prompt(context: str, foir: float, ltv: float) -> str:
    """Instruction prompt for Indian mortgage eligibility with reasons."""
    return f"""### Instruction:
You are a senior credit underwriter for Indian home loans, following RBI/NHB mortgage guidelines and common HFC practices.
Assess the applicant's HOME LOAN eligibility based on the context and calculated ratios. Provide a clear decision and reasons grounded in regulations and prudent underwriting.

### Input:
Applicant Context: {context}
Computed Ratios: FOIR={foir:.2f}, LTV={ltv:.2f}

Output format:
- Eligibility: Yes/No
- Reasons: concise bullet points referencing relevant constraints (e.g., KYC, age, FOIR affordability, LTV limits, credit score, employment stability)
- Risk flags: short list
- If Not eligible: specify what needs to improve

### Response:
"""

def format_simple_eligibility_prompt(context: str) -> str:
    """Prompt variant that uses calculator context without LTV/property inputs."""
    return f"""### Instruction:
You are a senior credit underwriter for Indian home loans. Using the provided affordability metrics and background,
decide whether the applicant is ELIGIBLE and justify in concise bullet points (FOIR affordability, income stability,
EMI headroom, potential risks). Keep the language crisp and professional.

### Input:
{context}

Output format:
- Eligibility: Yes/No
- Reasons: 3–6 bullets (affordability, obligations, risk flags)
- Suggestions: (optional) ways to improve approval odds

### Response:
"""
def main():
    # Header
    st.markdown('<div class="main-header">📋 Regulatory Compliance Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Fine-tuned Llama 3.1 8B with LoRA</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_path = st.text_input(
            "Model Path",
            value="./llama_regulatory_model",
            help="Path to the fine-tuned model directory"
        )
        
        base_model = st.text_input(
            "Base Model",
            value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            help="Hugging Face model name"
        )
        use_adapters = st.checkbox(
            "Use LoRA adapters (PEFT)",
            value=False,
            help="Enable to load fine-tuned LoRA adapters from the model path"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=1024,
            value=512,
            step=50,
            help="Maximum tokens to generate"
        )
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info("""
        This application demonstrates a fine-tuned Llama 3.1 8B model 
        specialized for regulatory compliance Q&A.
        
        **Features:**
        - LoRA fine-tuning (99% parameter reduction)
        - 4-bit quantization
        - Domain-specific knowledge
        """)
        
        # Check if model exists only when adapters are enabled
        if use_adapters and not os.path.exists(model_path):
            st.warning(f"⚠️ Adapters path not found: {model_path}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📚 Case Studies", "💬 Custom Query", "🧮 Eligibility"])
    
    # Tab 1: Home
    with tab1:
        st.header("Welcome to Regulatory Compliance Q&A Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Purpose")
            st.write("""
            This application demonstrates a fine-tuned language model specifically 
            trained on regulatory compliance documents including:
            - RBI Housing Finance regulations
            - KYC/AML requirements
            - Capital adequacy standards
            - Registration and licensing procedures
            """)
        
        with col2:
            st.subheader("🚀 Features")
            st.write("""
            - **Domain-Specific Knowledge**: Trained on regulatory documents
            - **Efficient Training**: LoRA + 4-bit quantization
            - **Accurate Responses**: Fine-tuned for compliance Q&A
            - **Case Study Testing**: Pre-built scenarios for validation
            """)
        
        st.divider()
        
        st.subheader("📊 Model Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Base Model", "Llama 3.1 8B")
        with col2:
            st.metric("Fine-tuning", "LoRA")
        with col3:
            st.metric("Parameters", "8M (0.1%)")
        with col4:
            st.metric("Quantization", "4-bit")
        
        st.divider()
        
        st.subheader("📖 How to Use")
        st.write("""
        1. **Case Studies Tab**: Test the model with pre-built regulatory scenarios
        2. **Custom Query Tab**: Ask your own regulatory compliance questions
        3. **Model Configuration**: Adjust settings in the sidebar
        """)
    
    # Tab 2: Case Studies
    with tab2:
        st.header("📚 Regulatory Compliance Case Studies")
        st.write("Test the model with real-world regulatory scenarios:")
        
        # Load model button
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.use_cuda = False
        
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model... This may take a few minutes."):
                model, tokenizer, use_cuda = load_model(model_path, base_model, use_adapters)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.use_cuda = use_cuda
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model. Please check the model path.")
        
        if st.session_state.model_loaded:
            st.success("✅ Model is ready!")
            
            # Case study selector
            selected_case = st.selectbox(
                "Select a Case Study:",
                list(CASE_STUDIES.keys())
            )
            
            case = CASE_STUDIES[selected_case]
            
            # Display case study (dedent to avoid markdown code blocks from indentation)
            scenario_md = textwrap.dedent(case["scenario"]).strip()
            st.markdown('<div class="case-study-box">', unsafe_allow_html=True)
            st.markdown(scenario_md)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Expected domain
            st.info(f"**Expected Domain:** {case['expected_domain']}")
            
            # Decide what to ask: eligibility decision vs general answer
            mode = st.radio(
                "Response Type",
                ["Eligibility decision with reasons", "General regulatory answer"],
                horizontal=True
            )
            # Generate answer button
            if st.button("🔍 Get Answer", type="primary"):
                with st.spinner("Generating answer..."):
                    if mode == "Eligibility decision with reasons":
                        prompt = format_case_study_eligibility_prompt(case["scenario"], case.get("question"))
                    else:
                        prompt = format_prompt(case["question"], context=textwrap.dedent(case["scenario"]).strip())
                    answer = generate_answer(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        prompt,
                        max_tokens,
                        st.session_state.use_cuda
                    )
                    
                    st.markdown(f'<div class="answer-box"><h4>📝 Model Answer:</h4><p>{answer}</p></div>', unsafe_allow_html=True)
                    
                    # Evaluation section
                    st.subheader("✅ Evaluation")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        relevance = st.radio(
                            "Relevance to Question:",
                            ["✅ Excellent", "✅ Good", "⚠️ Fair", "❌ Poor"],
                            key=f"relevance_{selected_case}"
                        )
                    
                    with col2:
                        accuracy = st.radio(
                            "Regulatory Accuracy:",
                            ["✅ Excellent", "✅ Good", "⚠️ Fair", "❌ Poor"],
                            key=f"accuracy_{selected_case}"
                        )
                    
                    with col3:
                        completeness = st.radio(
                            "Answer Completeness:",
                            ["✅ Excellent", "✅ Good", "⚠️ Fair", "❌ Poor"],
                            key=f"completeness_{selected_case}"
                        )
        else:
            st.warning("⚠️ Please load the model first using the button above.")
    
    # Tab 3: Custom Query
    with tab3:
        st.header("💬 Custom Regulatory Query")
        st.write("Ask your own regulatory compliance questions:")
        
        if st.session_state.model_loaded:
            # Question input
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., What are the capital adequacy requirements for housing finance companies?"
            )
            
            # Optional context
            context = st.text_area(
                "Optional Context (if any):",
                height=80,
                placeholder="Provide additional context if needed..."
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                generate_btn = st.button("🔍 Generate Answer", type="primary")
            
            if generate_btn and question:
                with st.spinner("Generating answer..."):
                    prompt = format_prompt(question, context if context else None)
                    answer = generate_answer(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        prompt,
                        max_tokens,
                        st.session_state.use_cuda
                    )
                    
                    st.markdown(f'<div class="answer-box"><h4>📝 Answer:</h4><p>{answer}</p></div>', unsafe_allow_html=True)
            elif generate_btn:
                st.warning("⚠️ Please enter a question first.")
        else:
            st.warning("⚠️ Please load the model first in the Case Studies tab.")

    # Tab 4: Eligibility
    with tab4:
        st.header("🧮 Calculate Home Loan Eligibility")
        st.caption("Simple calculator: we assume max affordable EMI = 40% of monthly income minus other EMIs.")

        # Sliders (styled similar to the provided example)
        gross_income = st.slider("Gross Income (Monthly)", min_value=10000, max_value=10000000, value=10000, step=1000, format="₹%d")
        tenure_years = st.slider("Tenure (Years)", min_value=1, max_value=30, value=30, step=1)
        interest_rate = st.slider("Interest Rate (% P.A.)", min_value=0.5, max_value=15.0, value=7.9, step=0.1)
        other_emis = st.slider("Other EMIs (Monthly)", min_value=0, max_value=10000000, value=0, step=1000, format="₹%d")

        # Calculation helpers
        def loan_from_emi(emi: float, annual_rate_percent: float, tenure_years: int) -> float:
            r = (annual_rate_percent / 100.0) / 12.0
            n = int(tenure_years * 12)
            if r <= 0 or n <= 0:
                return 0.0
            try:
                factor = ((1 + r) ** n - 1) / (r * (1 + r) ** n)
                return float(emi * factor)
            except Exception:
                return 0.0

        affordable_emi = max(gross_income * 0.40 - other_emis, 0)
        eligible_loan = loan_from_emi(affordable_emi, interest_rate, tenure_years)

        # Layout with right-side summary
        left, right = st.columns([2, 1])
        with left:
            st.write("")  # spacing
        with right:
            st.subheader("Your Home Loan Eligibility")
            st.markdown(f"### ₹{eligible_loan:,.0f}")
            st.caption("Your Home Loan EMI will be")
            st.markdown(f"### ₹{affordable_emi:,.0f} /month")
            # st.button("Apply Now")

        # Optional model-based decision and rationale
        st.divider()
        st.subheader("Model-based Decision (Optional)")
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if st.session_state.model_loaded:
            background = st.text_area(
                "Optional Background (employment, credit, property notes):",
                placeholder="e.g., Salaried 5 yrs, CIBIL 760, no defaults. Apartment purchase; adequate savings for down payment."
            )
            auto_run = st.checkbox("Auto-run model when sliders change", value=False)
            ask = st.button("Ask Model for Decision")
            should_run = ask or auto_run
            if should_run:
                foir = compute_foir(float(gross_income), float(other_emis), float(affordable_emi))
                context_lines = [
                    f"Monthly income: ₹{gross_income:,.0f}",
                    f"Other EMIs: ₹{other_emis:,.0f}",
                    f"Assumed affordable EMI (40% rule): ₹{affordable_emi:,.0f}",
                    f"Computed FOIR (incl. new EMI): {foir*100:.1f}%",
                    f"Tenure: {tenure_years} years",
                    f"Interest rate: {interest_rate:.2f}% p.a.",
                    f"Estimated eligible loan: ₹{eligible_loan:,.0f}",
                ]
                if background.strip():
                    context_lines.append(f"Background: {background.strip()}")
                ctx = "\n".join(context_lines)
                prompt = format_simple_eligibility_prompt(ctx)
                with st.spinner("Evaluating with model..."):
                    answer = generate_answer(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        prompt,
                        max_new_tokens=512,
                        use_cuda=st.session_state.use_cuda
                    )
                st.markdown(f'<div class="answer-box"><h4>📝 Model Decision:</h4><p>{answer}</p></div>', unsafe_allow_html=True)
        else:
            st.info("Load the model in the Case Studies tab to get a narrative decision. The calculator above works without the model.")

if __name__ == "__main__":
    main()

