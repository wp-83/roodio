#!/usr/bin/env bash
# ==============================================================================
#  Roodio - Start Script (Mac / Linux)
#  Run this after setup.sh is complete:
#    bash start.sh
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo ""
echo "========================================"
echo -e "${CYAN}  Roodio - Starting Local Servers${NC}"
echo "========================================"

# Cleanup function: stop all child processes on Ctrl+C
cleanup() {
    echo -e "\n${YELLOW}  Stopping all servers...${NC}"
    kill $(jobs -p) 2>/dev/null
    echo "  Done."
    exit 0
}
trap cleanup INT TERM

# Start Flask ML API in background
echo -e "\n${CYAN}[1/2] Starting Flask ML API (port 7860)...${NC}"
VENV_PYTHON="$REPO_ROOT/machineLearning/api/venv/bin/python3"
(cd "$REPO_ROOT/machineLearning/api" && "$VENV_PYTHON" app.py) &

sleep 3

# Start Laravel Webapp in background
echo -e "${GREEN}[2/2] Starting Laravel Webapp (port 8000)...${NC}"
(cd "$REPO_ROOT/webApp" && php artisan serve --host=0.0.0.0 --port=8000) &

echo ""
echo "========================================"
echo -e "${GREEN}  All servers are running!${NC}"
echo "========================================"
echo ""
echo -e "${CYAN}  Open in browser:${NC}"
echo "    Webapp   http://localhost:8000"
echo "    ML API   http://localhost:7860"
echo ""
echo -e "${YELLOW}  Press Ctrl+C to stop all servers.${NC}"
echo ""

# Wait for all background processes
wait
