# src/main.py

import sys
sys.path.append('.')

from utils import detect_plagiarism

if __name__ == "__main__":
    text1 = input("Enter the first text: ")
    text2 = input("Enter the second text: ")

    results = detect_plagiarism(text1, text2)
    print("\nPlagiarism Detection Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
