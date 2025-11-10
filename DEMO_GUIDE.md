# Demo Guide for Mentor Presentation

## Quick Start

### Option 1: Run Locally (macOS)
```bash
# Activate environment
source venv/bin/activate

# Install streamlit if needed
pip install streamlit

# Run the app
streamlit run frontend_app.py
# Or use the script:
./run_frontend.sh
```

### Option 2: Run on GCP VM (Recommended for GPU)
```bash
# SSH into VM
gcloud compute ssh llama-training --zone=us-east1-c --project=tcs-model-training

# On VM
source venv/bin/activate
pip install streamlit
streamlit run frontend_app.py --server.port 8501 --server.address 0.0.0.0
```

Then access: `http://YOUR_VM_IP:8501`

## Demo Flow (10-15 minutes)

### 1. Introduction (2 min)
- Show Home tab
- Explain the model: Llama 3.1 8B fine-tuned with LoRA
- Highlight efficiency: 99% parameter reduction, $5 training cost

### 2. Load Model (2-3 min)
- Go to Case Studies tab
- Click "Load Model" button
- Explain: Model loads with 4-bit quantization
- Wait for "Model loaded successfully" message

### 3. Test Case Studies (5-7 min)

**Case Study 1: Capital Adequacy**
- Select "Case Study 1: Capital Adequacy Compliance"
- Read the scenario
- Click "Get Answer"
- **Expected**: Model provides specific capital adequacy requirements
- **Evaluation**: Show how to rate the answer

**Case Study 2: KYC Update**
- Select "Case Study 2: KYC Update for Minor Account"
- Click "Get Answer"
- **Expected**: Model mentions fresh photographs and CDD documents
- **Highlight**: Domain-specific knowledge

**Case Study 3: Exposure Limits**
- Select "Case Study 3: Exposure Limit Compliance"
- Click "Get Answer"
- **Expected**: Model provides "5% of total assets"
- **Highlight**: Specific regulatory percentages

### 4. Custom Query (2-3 min)
- Go to Custom Query tab
- Enter: "What are the eligibility criteria for housing finance companies?"
- Click "Generate Answer"
- Show real-time generation

### 5. Q&A (2-3 min)
- Allow mentor to ask questions
- Test with their own regulatory questions
- Show model's ability to handle various queries

## Key Points to Emphasize

### Technical Achievements
- ✅ **99% Parameter Reduction**: Only 8M parameters trained (vs 8B)
- ✅ **88% Memory Reduction**: 6GB vs 52GB
- ✅ **67% Cost Savings**: $5 vs $15
- ✅ **67% Time Savings**: 8 hours vs 24 hours

### Model Quality
- ✅ **Domain-Specific**: Trained on regulatory documents
- ✅ **Accurate**: Provides correct regulatory information
- ✅ **Professional**: Uses proper terminology
- ✅ **Actionable**: Gives compliance guidance

### Use Cases
- Compliance teams can quickly get regulatory answers
- Reduces time spent searching through documents
- Ensures consistent regulatory interpretation
- Can be integrated into compliance workflows

## Troubleshooting

### If Model Doesn't Load
1. Check model path in sidebar
2. Verify Hugging Face authentication
3. Check console for error messages

### If Answers Are Slow
- On CPU: Normal (30-60 seconds)
- On GPU: Should be faster (5-10 seconds)
- First query after loading takes longer

### If Model Gives Generic Answers
- Ensure LoRA adapters are loaded correctly
- Check that model path points to fine-tuned model
- Verify training was completed successfully

## Backup Plan

If the live demo fails:
1. Have screenshots ready of:
   - Case study answers
   - Model configuration
   - Training results
2. Show the graphs from `paper_graphs/` folder
3. Demonstrate with `inference_example.py` script

## Expected Answers (Reference)

### Case Study 1: Capital Adequacy
Should mention specific capital adequacy ratios and regulatory framework.

### Case Study 2: KYC Update
Should mention: "Fresh photographs and CDD documents per current standards must be obtained."

### Case Study 3: Exposure Limits
Should provide: "5% of total assets"

### Case Study 4: Registration Cancellation
Should list conditions like:
- Ceases to carry on business
- Failed to comply with conditions
- Failed to fulfill requirements

### Case Study 5: Securities Market Value
Should mention: "Market value must not fall below the percentage of public deposits specified in Chapter III"

## Success Metrics

A successful demo should show:
- ✅ Model loads successfully
- ✅ Answers are relevant and accurate
- ✅ Model uses regulatory terminology
- ✅ Answers are complete and actionable
- ✅ Interface is user-friendly

## Post-Demo Discussion Points

1. **Scalability**: Can train on more documents
2. **Integration**: Can be integrated into existing systems
3. **Updates**: Easy to retrain with new regulations
4. **Cost**: Very cost-effective for organizations
5. **Accuracy**: Can be improved with more training data


