#!/usr/bin/env bash

set -e

print_help() {
    echo "Usage: $0 [option] [env_name]"
    echo "Options:"
    echo "  create      Create a new virtual environment (and install requirements.txt if present)"
    echo "  activate    Activate the virtual environment"
    echo "  install     Install dependencies from requirements.txt or setup.py"
    echo "  export      Export installed dependencies to requirements.txt"
    echo "  remove      Remove the virtual environment"
    echo "  -h, --help  Show this help message"
}

get_activate_path() {
    local env_name=${1:-".venv"}

    # Detect platform-specific activation script
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "$env_name/Scripts/activate"
    else
        echo "$env_name/bin/activate"
    fi
}

check_python() {
    if command -v python3 &> /dev/null && python3 --version &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null && python --version &> /dev/null; then
        echo "python"
    elif command -v py &> /dev/null && py --version &> /dev/null; then
        echo "py"
    else
        echo "Error: Python is not installed or not working." >&2
        exit 1
    fi
}

create_venv() {
    local env_name=${1:-".venv"}
    local python_cmd
    python_cmd=$(check_python)

    if [ -d "$env_name" ]; then
        echo "Virtual environment '$env_name' already exists. Aborting."
        return 1
    fi

    echo "Creating virtual environment: $env_name"
    "$python_cmd" -m venv "$env_name"

    local activate_path
    activate_path=$(get_activate_path "$env_name")
    # shellcheck disable=SC1090
    source "$activate_path"

    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        python -m pip install -r requirements.txt
    fi
}

activate_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found."
        return 1
    fi

    local activate_path
    activate_path=$(get_activate_path "$env_name")
    echo "Running the following command to activate your environment:"
    echo "source $activate_path"
    # activate the environment
    source "$activate_path" || {
        echo "Failed to activate the virtual environment. Please check the path."
        return 1
    }
    echo "Virtual environment '$env_name' activated."
}

install_deps() {
    local env_name=${1:-".venv"}
    local activate_path
    activate_path=$(get_activate_path "$env_name")

    if [ ! -f "$activate_path" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' first."
        return 1
    fi

    # shellcheck disable=SC1090
    source "$activate_path"

    if [ -f "requirements.txt" ]; then
        echo "Installing from requirements.txt..."
        pip install -r requirements.txt
    fi

    if [ -f "setup.py" ]; then
        echo "Installing from setup.py..."
        pip install -e .
    fi
}

export_deps() {
    local env_name=${1:-".venv"}
    local activate_path
    activate_path=$(get_activate_path "$env_name")

    if [ ! -f "$activate_path" ]; then
        echo "Virtual environment '$env_name' not found."
        return 1
    fi

    if [ -f "requirements.txt" ]; then
        echo "requirements.txt already exists. Please remove it first."
        return 1
    fi

    # shellcheck disable=SC1090
    source "$activate_path"
    pip freeze > requirements.txt
    echo "Dependencies exported to requirements.txt"
}

remove_venv() {
    local env_name=${1:-".venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found."
        return 1
    fi

    echo "Removing virtual environment '$env_name'..."
    rm -rf "$env_name"
}

# --------- Entry Point ---------

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_help
    exit 0
fi

case "$1" in
    "create")
        create_venv "$2"
        ;;
    "activate")
        activate_venv "$2"
        ;;
    "install")
        install_deps "$2"
        ;;
    "export")
        export_deps "$2"
        ;;
    "remove")
        remove_venv "$2"
        ;;
    *)
        echo "Unknown option: $1"
        print_help
        exit 1
        ;;
esac
