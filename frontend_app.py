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
    "Case Study 1: Young Couple Purchase": {
        "scenario": """
        **Scenario:** Arjun (34) and Neha (32) are salaried professionals in Mumbai with combined after-tax income of ₹1,20,000.
        They also earn ₹15,000/month from a rental unit. They currently pay a single EMI of ₹10,000.
        They plan to buy a ₹80,00,000 apartment and need a ₹60,00,000 loan for 25 years at 8.60% p.a.

        **Quick facts:**
        - Monthly income (after tax): ₹1,20,000
        - Additional income: ₹15,000
        - Existing EMIs: ₹10,000
        - Proposed loan tenure: 25 years
        - Interest rate: 8.60% p.a. (floating)
        - CIBIL scores: 782 / 770
        - Down payment available: ₹20,00,000 (25% margin)

        **Question:** Are Arjun and Neha likely eligible for the ₹60,00,000 home loan? Provide approval reasons
        and standard conditions (documentation, insurance, etc.).
        """,
        "question": "What are the capital adequacy requirements for housing finance companies?",
        "expected_domain": "Capital Adequacy, Regulatory Compliance"
    },
    "Case Study 2: High Obligations Borrower": {
        "scenario": """
        **Scenario:** Amit (37) is a salaried operations manager in Pune with after-tax income of ₹55,000.
        He has no secondary income and already pays EMIs totaling ₹20,000 (personal loan + car loan).
        He wants to buy a ₹55,00,000 apartment with a ₹50,00,000 loan for 20 years at 9.50% p.a.
        His savings for margin are ₹5,00,000 and his CIBIL score is 705.

        **Quick facts:**
        - Monthly income (after tax): ₹55,000
        - Additional income: ₹0
        - Existing EMIs: ₹20,000
        - Proposed loan tenure: 20 years
        - Interest rate: 9.50% p.a.
        - Property value: ₹55,00,000 (LTV ~91%)
        - Savings for margin: ₹5,00,000

        **Question:** Is Amit likely to be DECLINED for the ₹50,00,000 loan? Explain the FOIR, LTV, and risk factors,
        and what changes would make the case acceptable.
        """,
        "question": "What is required for KYC updates when a minor becomes an adult?",
        "expected_domain": "KYC/AML, Customer Due Diligence"
    },
    "Case Study 3: Self-Employed Entrepreneur": {
        "scenario": """
        **Scenario:** Kavya (31) runs a design studio in Bengaluru with average post-tax income of ₹90,000
        and additional freelance income of ₹10,000. She has one education loan EMI of ₹5,000.
        She wants a ₹45,00,000 loan for a ₹60,00,000 townhouse over 20 years at 9.20% p.a.
        Her CIBIL score is 748 with 3-year business vintage.

        **Quick facts:**
        - Monthly income (after tax): ₹90,000
        - Additional income: ₹10,000
        - Existing EMIs: ₹5,000
        - Proposed tenure: 20 years
        - Interest rate: 9.20% p.a.
        - Property value: ₹60,00,000 (LTV 75%)

        **Question:** Is Kavya likely to receive a CONDITIONAL approval? Highlight strengths (income, margin)
        and risk flags (self-employed stability, documentation) the lender will evaluate.
        """,
        "question": "What is the total exposure limit to housing finance and CRE-Residential Housing for all StCBs/DCCBs?",
        "expected_domain": "Risk Management, Exposure Limits"
    },
    "Case Study 4: Low CIBIL Borrower": {
        "scenario": """
        **Scenario:** Sunita (36) is a teacher in Delhi earning ₹2,00,000 after tax with no additional income.
        She pays ₹15,000 in existing EMIs. She wants a ₹80,00,000 loan on a ₹1,20,00,000 apartment for 20 years at 9.30% p.a.
        Her CIBIL score is 640 due to a past 90+ dpd credit card that was settled 12 months ago.

        **Quick facts:**
        - Monthly income (after tax): ₹2,00,000
        - Additional income: ₹0
        - Existing EMIs: ₹15,000
        - Proposed tenure: 20 years
        - Interest rate: 9.30% p.a.
        - Property value: ₹1,20,00,000 (LTV 67%)
        - Credit history: CIBIL 640; major delinquency closed 12 months back

        **Question:** Will Sunita likely face a DECLINE despite comfortable FOIR/LTV? Provide reasons tied to credit
        history and list remediation (seasoning, guarantor, bureau improvement) required for future approval.
        """,
        "question": "Under what conditions can the Reserve Bank cancel a certificate of registration for a housing finance institution?",
        "expected_domain": "Registration, Regulatory Enforcement"
    },
    "Case Study 5: Early Career Applicant": {
        "scenario": """
        **Scenario:** Rahul (29) is a salaried analyst in Hyderabad with after-tax income of ₹45,000 and additional freelance
        income of ₹5,000. He has no existing EMIs. He plans to purchase a ₹40,00,000 apartment with a ₹35,00,000 loan for 25 years
        at 8.90% p.a. He joined his current job 6 months ago but has 3 years total experience. CIBIL score is 760.

        **Quick facts:**
        - Monthly income (after tax): ₹45,000
        - Additional income: ₹5,000
        - Existing EMIs: ₹0
        - Proposed tenure: 25 years
        - Interest rate: 8.90% p.a.
        - Property value: ₹40,00,000 (LTV 87.5%)
        - Employment stability: 6 months in current role

        **Question:** Is Rahul likely to get a CONDITIONAL approval or DECLINE? Explain how FOIR, high LTV, and employment
        stability influence the decision and list typical lender conditions for such applicants.
        """,
        "question": "What is the market value requirement for securities held for depositors?",
        "expected_domain": "Securities, Deposit Protection"
    },
    "Case Study 6: Definite Decline": {
        "scenario": """
        **Scenario:** Nikhil (35) is a salaried sales executive in Nagpur with after-tax income of ₹45,000/month.
        He has no additional income and already pays EMIs totaling ₹15,000 (personal loan + bike loan).
        He wants a ₹45,00,000 home loan for 20 years at 10.50% p.a. to buy a ₹50,00,000 apartment (LTV 90%).
        His CIBIL score is 620 with recent 60+ dpd on a credit card.

        **Quick facts:**
        - Monthly income (after tax): ₹45,000
        - Additional income: ₹0
        - Existing EMIs: ₹15,000
        - Proposed tenure: 20 years
        - Interest rate: 10.50% p.a.
        - Property value: ₹50,00,000 (requested LTV 90%)
        - Requested loan: ₹45,00,000
        - Credit history: CIBIL 620; recent delinquency

        **Question:** Based on FOIR affordability, high LTV vs typical caps (≤80–90% by ticket), and low CIBIL,
        is this case NOT ELIGIBLE? Provide a clear “Not eligible” decision with reasons and specific improvements
        required (e.g., higher margin to lower LTV, close obligations to reduce FOIR, bureau repair and seasoning).
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

def build_calculator_reasoning(
    gross_income: float,
    other_emis: float,
    affordable_emi: float,
    foir: float,
    tenure_years: int,
    effective_rate: float,
    eligible_loan: float,
    expected_loan: float,
    use_ltv: bool,
    property_value: float,
    ltv_cap_percent: int,
) -> tuple[str, bool]:
    """
    Create a deterministic explanation aligned with the calculator output.
    Returns (reasoning_markdown, is_eligible_bool)
    """
    reasons = []
    # Core decision
    is_eligible = expected_loan <= eligible_loan if expected_loan and expected_loan > 0 else True
    # FOIR narrative
    reasons.append(f"- Income considered: ₹{gross_income:,.0f}/month; existing EMIs: ₹{other_emis:,.0f}/month.")
    reasons.append(f"- Affordability applied: max EMI ≈ ₹{affordable_emi:,.0f} (policy FOIR rule).")
    reasons.append(f"- Computed FOIR with new EMI: {foir*100:.1f}% (must be within policy limit).")
    # Tenure / rate narrative
    reasons.append(f"- Tenure used: {tenure_years} years; Interest rate used for eligibility: {effective_rate:.2f}% p.a.")
    # LTV narrative
    if use_ltv and property_value > 0:
        ltv_actual = (expected_loan / property_value) if expected_loan else 0.0
        reasons.append(f"- LTV check: requested LTV {ltv_actual*100:.1f}% vs cap {ltv_cap_percent}% "
                       f"(property value ₹{property_value:,.0f}).")
        if expected_loan and ltv_actual*100 > ltv_cap_percent:
            is_eligible = False
            reasons.append("- Requested loan breaches LTV cap → requires higher down payment.")
    # Requested vs eligible
    if expected_loan and expected_loan > 0:
        shortfall = max(expected_loan - eligible_loan, 0)
        reasons.append(f"- Calculator eligibility ≈ ₹{eligible_loan:,.0f}; Requested loan = ₹{expected_loan:,.0f}.")
        if shortfall > 0:
            reasons.append(f"- Shortfall of ≈ ₹{shortfall:,.0f} compared to requested amount.")
            is_eligible = False
    # Wrap up with suggestions
    if not is_eligible:
        reasons.append("- Improve chances by increasing margin (lower LTV), closing obligations to reduce FOIR, "
                       "or choosing longer tenure (if policy allows).")
    else:
        reasons.append("- Requested loan is within the calculator’s eligibility and policy parameters.")
    md = "**Eligibility:** " + ("No" if not is_eligible else "Yes") + "\n\n**Reasons:**\n" + "\n".join(reasons)
    return md, is_eligible

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
        st.caption("Simple calculator tuned to match common bank calculators (default FOIR 50% of gross income).")

        # Sliders (styled similar to the provided example)
        gross_income = st.slider("Gross Income (Monthly)", min_value=10000, max_value=10000000, value=10000, step=1000, format="₹%d")
        tenure_years = st.slider("Tenure (Years)", min_value=1, max_value=30, value=30, step=1)
        interest_rate = st.slider("Interest Rate (% P.A.)", min_value=0.5, max_value=15.0, value=7.9, step=0.1)
        other_emis = st.slider("Other EMIs (Monthly)", min_value=0, max_value=10000000, value=0, step=1000, format="₹%d")
        expected_loan = st.number_input("Expected Loan Amount (₹)", min_value=0.0, value=0.0, step=50000.0, format="%.0f")

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

        # Advanced policy knobs (defaults chosen to match HDFC-style calculator)
        with st.expander("Advanced settings (policy assumptions)", expanded=False):
            foir_limit = st.slider("FOIR Limit (%)", 30, 60, 50, 1, help="Maximum share of income allowed for total EMIs")
            net_income_factor = st.slider("Net Income Factor (after tax/EPF) (%)", 50, 100, 100, 1, help="Use 100% to mimic gross-income based calculators")
            stress_buffer = st.slider("Stress Rate Buffer (+% p.a.)", 0.0, 3.0, 0.0, 0.1, help="Add buffer over offered rate (many banks use 0–2%)")
            use_ltv = st.checkbox("Apply LTV cap", value=False)
            property_value = st.number_input("Property Value (₹)", min_value=0.0, value=0.0, step=50000.0, format="%.2f")
            ltv_cap_percent = st.slider("LTV Cap (%)", 50, 90, 80, 1)
            use_age_cap = st.checkbox("Cap tenure by retirement age", value=False)
            col_age1, col_age2 = st.columns(2)
            with col_age1:
                age = st.number_input("Applicant Age", min_value=18, max_value=80, value=30)
            with col_age2:
                retirement_age = st.number_input("Retirement Age", min_value=50, max_value=70, value=60)

        # Compute adjusted inputs
        net_income = gross_income * (net_income_factor / 100.0)
        max_emi_allowed = max((foir_limit / 100.0) * net_income - other_emis, 0)
        effective_rate = interest_rate + stress_buffer
        tenure_effective = tenure_years
        if use_age_cap:
            tenure_effective = min(tenure_years, max(1, retirement_age - age))

        affordable_emi = max_emi_allowed
        eligible_loan = loan_from_emi(affordable_emi, effective_rate, tenure_effective)
        if use_ltv and property_value > 0:
            eligible_loan = min(eligible_loan, property_value * (ltv_cap_percent / 100.0))

        # Round to typical display rules
        eligible_loan = (eligible_loan // 1)  # drop paise
        eligible_loan = (eligible_loan // 1)  # keep rupees for precision

        # Layout with right-side summary
        left, right = st.columns([2, 1])
        with left:
            st.write("")  # spacing
        with right:
            st.subheader("Your Home Loan Eligibility")
            st.markdown(f"### ₹{eligible_loan:,.0f}")
            st.caption("Your Home Loan EMI will be")
            # round EMI to nearest hundred to mimic many bank UIs
            emi_display = (affordable_emi // 100) * 100
            st.markdown(f"### ₹{emi_display:,.0f} /month")
            # Decision based on user's expected loan
            if expected_loan > 0:
                if expected_loan <= eligible_loan:
                    st.success(f"Eligibility Decision: ELIGIBLE for requested ₹{expected_loan:,.0f}")
                else:
                    st.error(f"Eligibility Decision: NOT ELIGIBLE for requested ₹{expected_loan:,.0f} "
                             f"(max ≈ ₹{eligible_loan:,.0f})")
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
            ask = st.button("Verify with Model")
            should_run = ask or auto_run
            if should_run:
                foir = compute_foir(float(gross_income), float(other_emis), float(affordable_emi))
                # Build deterministic reasoning aligned with calculator output
                reasoning_md, ok = build_calculator_reasoning(
                    gross_income=float(gross_income),
                    other_emis=float(other_emis),
                    affordable_emi=float(affordable_emi),
                    foir=float(foir),
                    tenure_years=int(tenure_effective),
                    effective_rate=float(effective_rate),
                    eligible_loan=float(eligible_loan),
                    expected_loan=float(expected_loan or 0),
                    use_ltv=use_ltv,
                    property_value=float(property_value or 0),
                    ltv_cap_percent=int(ltv_cap_percent),
                )
                if expected_loan and expected_loan > 0:
                    if ok:
                        st.success(f"Model Decision: ELIGIBLE for requested ₹{expected_loan:,.0f}")
                    else:
                        st.error(f"Model Decision: NOT ELIGIBLE for requested ₹{expected_loan:,.0f} "
                                 )
                st.markdown(f'<div class="answer-box"><h4>📝 Reasoning:</h4><p>{reasoning_md}</p></div>', unsafe_allow_html=True)
        else:
            st.info("Load the model in the Case Studies tab to get a narrative decision. The calculator above works without the model.")

if __name__ == "__main__":
    main()

