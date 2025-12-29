#!/bin/bash
# Setup script for annotation environment

echo "=========================================="
echo "ANNOTATION SETUP"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "[OK] Created .env file"
    echo ""
    echo "[WARNING]  IMPORTANT: Edit .env and add your OpenAI API key"
    echo "   Get your key from: https://platform.openai.com/api-keys"
    echo ""
    read -p "Press Enter to open .env in editor..."
    ${EDITOR:-nano} .env
else
    echo "[OK] .env file already exists"
fi

# Verify API key is set
if grep -q "your_api_key_here" .env; then
    echo ""
    echo "[ERROR] WARNING: API key not configured!"
    echo "   Please edit .env and add your real OpenAI API key"
    exit 1
fi

echo ""
echo "[OK] API key configured"
echo ""

# Export the key for current session
export $(cat .env | xargs)

# Test API connection
echo "Testing OpenAI API connection..."
python3 << 'EOF'
import os
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("[ERROR] ERROR: OPENAI_API_KEY not found in environment")
    exit(1)

try:
    client = OpenAI(api_key=api_key)
    # Simple test call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API working'"}],
        max_tokens=10
    )
    print("[OK] API connection successful!")
    print(f"  Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"[ERROR] API connection failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SETUP COMPLETE"
    echo "=========================================="
    echo ""
    echo "You can now run the annotation script:"
    echo "  python3 scripts/automated_annotation.py"
    echo ""
    echo "Or test the prompts first:"
    echo "  python3 scripts/test_annotation_prompts.py"
    echo ""
else
    echo ""
    echo "[ERROR] Setup failed. Please check your API key and try again."
    exit 1
fi
