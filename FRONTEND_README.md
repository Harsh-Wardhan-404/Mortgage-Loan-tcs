# Frontend Application - Regulatory Compliance Q&A

This Streamlit application provides a user-friendly interface to test the fine-tuned regulatory compliance model with case studies.

## Features

- 🏠 **Home Page**: Overview of the model and its capabilities
- 📚 **Case Studies**: Pre-built regulatory scenarios for testing
- 💬 **Custom Query**: Ask your own regulatory questions
- ⚙️ **Configuration**: Adjustable model settings

## Installation

1. **Install dependencies:**
```bash
pip install streamlit>=1.28.0
# Or install all requirements
pip install -r requirements.txt
```

2. **Ensure model is available:**
   - The model should be in `./llama_regulatory_model/` directory
   - Or update the model path in the sidebar

## Running the Application

### Local Machine (macOS/CPU)

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run frontend_app.py
```

The app will open in your browser at `http://localhost:8501`

### On GCP VM (GPU)

```bash
# SSH into your VM
gcloud compute ssh llama-training --zone=us-east1-c --project=tcs-model-training

# Activate virtual environment
source venv/bin/activate

# Install streamlit if not already installed
pip install streamlit

# Run the app
streamlit run frontend_app.py --server.port 8501 --server.address 0.0.0.0
```

Then access via: `http://YOUR_VM_EXTERNAL_IP:8501`

**Note:** You may need to create a firewall rule for port 8501:
```bash
gcloud compute firewall-rules create allow-streamlit-8501 \
    --allow tcp:8501 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow Streamlit access"
```

## Usage

### Step 1: Load the Model
1. Open the **Case Studies** tab
2. Click **"🔄 Load Model"** button
3. Wait for the model to load (may take a few minutes on first load)

### Step 2: Test with Case Studies
1. Select a case study from the dropdown
2. Read the scenario
3. Click **"🔍 Get Answer"** to see the model's response
4. Evaluate the answer using the rating system

### Step 3: Custom Queries
1. Go to the **Custom Query** tab
2. Enter your regulatory question
3. Optionally add context
4. Click **"🔍 Generate Answer"**

## Case Studies Included

1. **Capital Adequacy Compliance** - Testing capital requirements knowledge
2. **KYC Update for Minor Account** - Testing KYC/AML knowledge
3. **Exposure Limit Compliance** - Testing risk management knowledge
4. **Registration Cancellation** - Testing regulatory enforcement knowledge
5. **Securities Market Value** - Testing deposit protection knowledge

See `CASE_STUDIES.md` for detailed case study descriptions.

## Configuration

Adjust settings in the sidebar:
- **Model Path**: Path to fine-tuned model directory
- **Base Model**: Hugging Face model name
- **Max Tokens**: Maximum tokens to generate (100-1024)

## Troubleshooting

### Model Not Loading
- Check that the model path is correct
- Ensure you have Hugging Face authentication set up
- Verify model files exist in the specified directory

### Slow Performance
- On CPU: Expect slower inference (30-60 seconds per query)
- On GPU: Should be faster (5-10 seconds per query)
- First load takes longer due to model loading

### Memory Issues
- On CPU: Ensure you have at least 16GB RAM
- On GPU: Model uses ~6GB VRAM with 4-bit quantization

## For Presentation/Demo

1. **Prepare in advance:**
   - Load the model before the presentation
   - Test all case studies to ensure they work
   - Have backup screenshots ready

2. **Demo flow:**
   - Start with Home tab (show overview)
   - Go to Case Studies tab
   - Load model (if not already loaded)
   - Run 2-3 case studies
   - Show Custom Query tab
   - Answer a question from the audience

3. **Key points to highlight:**
   - Model provides domain-specific answers
   - Uses proper regulatory terminology
   - Handles complex scenarios
   - Fast inference (on GPU)

## Screenshots for Documentation

The app includes:
- Professional UI with custom styling
- Clear case study scenarios
- Evaluation metrics
- Model information display


