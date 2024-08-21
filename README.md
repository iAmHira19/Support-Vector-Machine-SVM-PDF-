# Support Vector Machines (SVM) Project

## Overview

This project demonstrates the application of Support Vector Machines (SVM) for classification tasks. It includes code examples, explanations of SVM concepts, and practical implementations using different kernels. This project is designed for educational purposes and provides a comprehensive overview of SVM, including linear and non-linear classification.

## Key Concepts

- **SVM Basics**: Aims to find the optimal hyperplane that maximizes the margin between classes.
- **Kernel Trick**: Allows SVM to handle non-linearly separable data by mapping it to a higher-dimensional space.
- **Types of SVM**: Linear SVM, Polynomial Kernel, Radial Basis Function (RBF) Kernel, Sigmoid Kernel.
- **Soft Margin vs. Hard Margin**: Balances between margin maximization and classification errors.
- **Hinge Loss**: Loss function used in SVM to train the model.

## Installation

To run the code in this project, you need Python and some specific libraries. Follow these steps to set up your environment:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/svm-project.git
   cd svm-project
   ```

2. **Create a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should contain:
   ```
   numpy
   pandas
   scikit-learn
   matplotlib
   ```

## Usage

1. **Linear SVM Example**
   ```python
   from sklearn import datasets
   from sklearn.svm import SVC
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   # Load dataset
   iris = datasets.load_iris()
   X, y = iris.data, iris.target

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Train SVM
   model = SVC(kernel='linear')
   model.fit(X_train, y_train)

   # Predict and evaluate
   y_pred = model.predict(X_test)
   print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
   ```

2. **Non-Linear SVM with RBF Kernel**
   ```python
   from sklearn.svm import SVC
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   # Create a dataset
   X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Train SVM with RBF kernel
   model = SVC(kernel='rbf', gamma='auto')
   model.fit(X_train, y_train)

   # Predict and evaluate
   y_pred = model.predict(X_test)
   print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
   ```

## File Structure

- `main.py`: Contains the main code examples and implementations.
- `requirements.txt`: Lists the required Python libraries.
- `README.md`: This file.

## Contributing

Feel free to submit issues, suggestions, or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, contact [yourname@example.com](mailto:yourname@example.com).

