# Age and Gender Prediction Using UTKFace Dataset

This project aims to predict age and gender using the UTKFace dataset. The model is built using TensorFlow and Keras and leverages convolutional neural networks (CNNs) for image processing and classification.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Installation

1. Install the required packages using pip:
    ```bash
    pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
    ```

2. Download the UTKFace dataset and place it in the project directory.

### Dataset

The UTKFace dataset contains images of faces with labels for age and gender. Each file name contains the age, gender, and other attributes in the format: `age_gender_race_date.jpg`.

### Project Structure

- `intial_model.ipynb`: Jupyter notebook containing the code for the project.
- `UTKFace/`: Directory containing the UTKFace dataset.
- `images.npy`, `ages.npy`, `genders.npy`, `lables.npy`: Numpy arrays of images, ages, genders, and combined labels.

### Code Overview

1. **Import Libraries**
    - Import necessary libraries for image processing, machine learning, and visualization.

2. **Set Directory Path**
    - Define the directory where the dataset is located.

3. **Initialize Lists and Read Data**
    - Initialize lists to store images, ages, and genders.
    - Loop through files to read images and extract age and gender information.

4. **Display Sample Image**
    - Display a sample image along with its age and gender.

5. **Convert Lists to NumPy Arrays**
    - Convert the lists to NumPy arrays for easier manipulation and processing.

6. **Save Data**
    - Save the NumPy arrays as `.npy` files.

7. **Plot Gender and Age Distribution**
    - Plot the distribution of genders and ages in the dataset.

8. **Create Labels Array**
    - Create an array combining ages and genders.

9. **Normalize Images**
    - Normalize image pixel values to the range [0, 1].

10. **Split Data into Training and Testing Sets**
    - Split the data into training and testing sets.

11. **Separate Age and Gender Labels**
    - Split the labels into separate arrays for age and gender.

12. **Define Convolutional Block**
    - Define a function for creating a convolutional block.

13. **Define the Model Architecture**
    - Define the model architecture using convolutional and dense layers.

14. **Instantiate and Summarize the Model**
    - Create an instance of the model and print its summary.

15. **Define Callbacks for Model Training**
    - Define callbacks for checkpointing and early stopping during training.

16. **Train the Model**
    - Train the model using the training data and validate on the test data.

17. **Evaluate the Model**
    - Evaluate the model on the test data.

18. **Make Predictions**
    - Use the trained model to make predictions on the test data.

19. **Plot Training History**
    - Plot the training history for loss and accuracy.

20. **Plot Actual vs Predicted Ages**
    - Plot the actual versus predicted ages.

21. **Generate Classification Report and Confusion Matrix**
    - Generate a classification report and confusion matrix for gender prediction.

22. **Test Model on a Single Image**
    - Test the model's prediction on individual images.

### Results

- The model can predict age and gender with reasonable accuracy.
- Visualizations of the training history, actual vs predicted ages, and confusion matrix are included.

### Acknowledgments

- The UTKFace dataset is available [here](https://www.kaggle.com/datasets/jangedoo/utkface-new).

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
