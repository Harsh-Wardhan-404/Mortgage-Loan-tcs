#!/bin/bash

################################################################################
# Run Frontend Application
# Starts the Streamlit frontend for regulatory compliance Q&A
################################################################################

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Regulatory Compliance Q&A Frontend...${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}Warning: Virtual environment not found. Using system Python.${NC}"
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}Streamlit not found. Installing...${NC}"
    pip install streamlit>=1.28.0
fi

# Check if model directory exists
if [ ! -d "llama_regulatory_model" ]; then
    echo -e "${YELLOW}Warning: Model directory 'llama_regulatory_model' not found.${NC}"
    echo -e "${YELLOW}You can specify a different path in the app sidebar.${NC}"
fi

# Run Streamlit
echo -e "${GREEN}Starting Streamlit application...${NC}"
echo -e "${GREEN}The app will open in your browser at http://localhost:8501${NC}"
echo ""

streamlit run frontend_app.py


