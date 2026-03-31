#!/bin/bash

# Public Service Navigation Assistant Setup Script
# This script sets up and runs the complete voice-enabled AI system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if ! command_exists curl; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_status "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites are installed!"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_warning "Created .env file from template. Please edit it with your configuration."
        else
            print_error "env.example file not found!"
            exit 1
        fi
    else
        print_status ".env file already exists."
    fi
    
    # Create necessary directories
    mkdir -p backend/data backend/vectorstore
    mkdir -p rasa/models rasa/logs
    mkdir -p voice/logs
    
    print_success "Environment setup complete!"
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build backend image
    print_status "Building backend image..."
    docker build -t public-service-backend ./backend
    
    # Build Rasa image
    print_status "Building Rasa image..."
    docker build -t public-service-rasa ./rasa
    
    # Build voice handler image
    print_status "Building voice handler image..."
    docker build -t public-service-voice ./voice
    
    print_success "All Docker images built successfully!"
}

# Function to start services
start_services() {
    print_status "Starting services with Docker Compose..."
    
    # Start Ollama first (for LLM)
    print_status "Starting Ollama LLM service..."
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    print_status "Waiting for Ollama to be ready..."
    sleep 30
    
    # Start all other services
    print_status "Starting all services..."
    docker-compose up -d
    
    print_success "All services started!"
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    local services=("backend" "rasa" "voice-handler")
    local ports=(8000 5005 5001)
    
    for i in "${!services[@]}"; do
        local service=${services[$i]}
        local port=${ports[$i]}
        
        print_status "Checking $service on port $port..."
        
        if curl -f http://localhost:$port/health >/dev/null 2>&1; then
            print_success "$service is healthy!"
        else
            print_warning "$service health check failed. It may still be starting up."
        fi
    done
}

# Function to train Rasa model
train_rasa() {
    print_status "Training Rasa model..."
    
    # Wait for Rasa service to be ready
    sleep 10
    
    # Train the model
    docker-compose exec rasa rasa train
    
    print_success "Rasa model training complete!"
}

# Function to show usage information
show_usage() {
    echo "Public Service Navigation Assistant"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Complete setup (check prerequisites, build images, start services)"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  build     - Build Docker images"
    echo "  health    - Check service health"
    echo "  train     - Train Rasa model"
    echo "  logs      - Show service logs"
    echo "  clean     - Clean up containers and images"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup    # Complete setup"
    echo "  $0 start    # Start services"
    echo "  $0 logs     # View logs"
}

# Function to show logs
show_logs() {
    print_status "Showing service logs..."
    docker-compose logs -f
}

# Function to clean up
cleanup() {
    print_status "Cleaning up containers and images..."
    
    docker-compose down -v
    docker system prune -f
    
    print_success "Cleanup complete!"
}

# Main script logic
case "${1:-setup}" in
    setup)
        print_status "Starting complete setup..."
        check_prerequisites
        setup_environment
        build_images
        start_services
        sleep 30
        check_health
        train_rasa
        print_success "Setup complete! Your Public Service Navigation Assistant is ready."
        print_status "Access the API at: http://localhost:8000"
        print_status "Rasa webhook at: http://localhost:5005"
        print_status "Voice handler at: http://localhost:5001"
        ;;
    start)
        start_services
        ;;
    stop)
        print_status "Stopping services..."
        docker-compose down
        print_success "Services stopped!"
        ;;
    restart)
        print_status "Restarting services..."
        docker-compose restart
        print_success "Services restarted!"
        ;;
    build)
        build_images
        ;;
    health)
        check_health
        ;;
    train)
        train_rasa
        ;;
    logs)
        show_logs
        ;;
    clean)
        cleanup
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 