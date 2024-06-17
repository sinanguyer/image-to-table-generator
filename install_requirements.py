import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    with open('requirements.txt', 'r') as file:
        packages = file.readlines()
    for package in packages:
        package = package.strip()
        if package:
            install(package)
    print("All packages installed successfully.")

if __name__ == "__main__":
    main()
