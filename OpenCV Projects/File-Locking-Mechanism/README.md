# File-Locking-Mechanism-Using-Face-Recognition

Easily secure your files and folders on Windows using Face Recognition technology. This project allows you to lock and unlock any folder with your unique face, providing an intuitive and secure way to protect your data.

## Features:
Train the system to recognize your face.
Lock and unlock any folder using face recognition.
Seamlessly integrates with Windows for quick file security.

## Steps to Lock a Folder on Your Windows System

### 1. Change directory:
```bash
   cd File-Locking-Mechanism
```
### 2. Train the System to Recognize Your Face:

Open the terminal and navigate to the repository folder. Run the following command to train the system with your face data:   

```bash
   python "Face Recognition.py" -train
```

### 3. Update the `Locker.bat` File:
   - Open the `Locker.bat` file using any text editor (e.g., Notepad).
   - Locate the `PASS` variable.
   - Replace the existing value with the **name** you used while training your face in the previous step.

### 4. Create a Locker Folder:
   Run the `Locker.bat` file by double-clicking it. This will create a folder named `Locker` in the current directory.

### 5. Store Your Files:
   Move any files or folders you want to secure into the `Locker` folder.

### 6. Lock and Unlock Your Files:
   - To **lock** the folder: Run the `Locker.bat` file again, and the folder will be locked.
   - To **unlock** the folder: Run the `Locker.bat` file, and the system will use face recognition to verify your identity. Upon successful recognition, the folder will unlock.

## How It Works:

- **Face Recognition**:  
   The system uses the **Local Binary Pattern Histogram (LBPH)** algorithm for face recognition. After training, it matches the captured face with the stored images during the locking/unlocking process.

- **Security**:  
   Only the user whose face was trained can unlock the folder, ensuring a personalized layer of protection.

## Troubleshooting:

- **Training Issues**:  
   If the training process fails, ensure your webcam is working properly and capture at least **100 images** with varying expressions.

- **Unlocking Issues**:  
   Make sure you provide good lighting conditions while using face recognition for unlocking the folder. If recognition fails repeatedly, retrain the model for better accuracy.

## Future Improvements:

- **Additional Algorithms**:  
   Incorporate algorithms like **Eigenfaces** or **Fisherfaces** for better accuracy.

- **Multi-User Support**:  
   Add support for multiple users to enable shared access to the locked folder.







