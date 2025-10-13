{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torchvision import datasets, transforms, models  # datsets  , transforms\n",
    "from torch.utils.data.sampler import SubsetRandomSampler\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 3;\n                var nbb_unformatted_code = \"%load_ext nb_black\";\n                var nbb_formatted_code = \"%load_ext nb_black\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%load_ext nb_black"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b> Dataset Link (Plant Vliiage Dataset ):</b><br> <a href='https://data.mendeley.com/datasets/tywbtsjrjv/1'> https://data.mendeley.com/datasets/tywbtsjrjv/1 </a> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 4;\n                var nbb_unformatted_code = \"transform = transforms.Compose(\\n    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]\\n)\";\n                var nbb_formatted_code = \"transform = transforms.Compose(\\n    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]\\n)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "transform = transforms.Compose(\n",
    "    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 5;\n                var nbb_unformatted_code = \"dataset = datasets.ImageFolder(\\\"Dataset\\\", transform=transform)\";\n                var nbb_formatted_code = \"dataset = datasets.ImageFolder(\\\"Dataset\\\", transform=transform)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "dataset = datasets.ImageFolder(\"Dataset\", transform=transform)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Dataset ImageFolder\n",
       "    Number of datapoints: 61486\n",
       "    Root Location: Dataset\n",
       "    Transforms (if any): Compose(\n",
       "                             Resize(size=255, interpolation=PIL.Image.BILINEAR)\n",
       "                             CenterCrop(size=(224, 224))\n",
       "                             ToTensor()\n",
       "                         )\n",
       "    Target Transforms (if any): None"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 6;\n                var nbb_unformatted_code = \"dataset\";\n                var nbb_formatted_code = \"dataset\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 7;\n                var nbb_unformatted_code = \"indices = list(range(len(dataset)))\";\n                var nbb_formatted_code = \"indices = list(range(len(dataset)))\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "indices = list(range(len(dataset)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 8;\n                var nbb_unformatted_code = \"split = int(np.floor(0.85 * len(dataset)))  # train_size\";\n                var nbb_formatted_code = \"split = int(np.floor(0.85 * len(dataset)))  # train_size\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "split = int(np.floor(0.85 * len(dataset)))  # train_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 9;\n                var nbb_unformatted_code = \"validation = int(np.floor(0.70 * split))  # validation\";\n                var nbb_formatted_code = \"validation = int(np.floor(0.70 * split))  # validation\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "validation = int(np.floor(0.70 * split))  # validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 36584 52263 61486\n"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 10;\n                var nbb_unformatted_code = \"print(0, validation, split, len(dataset))\";\n                var nbb_formatted_code = \"print(0, validation, split, len(dataset))\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(0, validation, split, len(dataset))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "length of train size :36584\n",
      "length of validation size :15679\n",
      "length of test size :24902\n"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 11;\n                var nbb_unformatted_code = \"print(f\\\"length of train size :{validation}\\\")\\nprint(f\\\"length of validation size :{split - validation}\\\")\\nprint(f\\\"length of test size :{len(dataset)-validation}\\\")\";\n                var nbb_formatted_code = \"print(f\\\"length of train size :{validation}\\\")\\nprint(f\\\"length of validation size :{split - validation}\\\")\\nprint(f\\\"length of test size :{len(dataset)-validation}\\\")\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(f\"length of train size :{validation}\")\n",
    "print(f\"length of validation size :{split - validation}\")\n",
    "print(f\"length of test size :{len(dataset)-validation}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 12;\n                var nbb_unformatted_code = \"np.random.shuffle(indices)\";\n                var nbb_formatted_code = \"np.random.shuffle(indices)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "np.random.shuffle(indices)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Split into Train and Test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 13;\n                var nbb_unformatted_code = \"train_indices, validation_indices, test_indices = (\\n    indices[:validation],\\n    indices[validation:split],\\n    indices[split:],\\n)\";\n                var nbb_formatted_code = \"train_indices, validation_indices, test_indices = (\\n    indices[:validation],\\n    indices[validation:split],\\n    indices[split:],\\n)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_indices, validation_indices, test_indices = (\n",
    "    indices[:validation],\n",
    "    indices[validation:split],\n",
    "    indices[split:],\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 14;\n                var nbb_unformatted_code = \"train_sampler = SubsetRandomSampler(train_indices)\\nvalidation_sampler = SubsetRandomSampler(validation_indices)\\ntest_sampler = SubsetRandomSampler(test_indices)\";\n                var nbb_formatted_code = \"train_sampler = SubsetRandomSampler(train_indices)\\nvalidation_sampler = SubsetRandomSampler(validation_indices)\\ntest_sampler = SubsetRandomSampler(test_indices)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_sampler = SubsetRandomSampler(train_indices)\n",
    "validation_sampler = SubsetRandomSampler(validation_indices)\n",
    "test_sampler = SubsetRandomSampler(test_indices)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 15;\n                var nbb_unformatted_code = \"targets_size = len(dataset.class_to_idx)\";\n                var nbb_formatted_code = \"targets_size = len(dataset.class_to_idx)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "targets_size = len(dataset.class_to_idx)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Convolution Aithmetic Equation : </b>(W - F + 2P) / S + 1 <br>\n",
    "W = Input Size<br>\n",
    "F = Filter Size<br>\n",
    "P = Padding Size<br>\n",
    "S = Stride <br>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Transfer Learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model = models.vgg16(pretrained=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# for params in model.parameters():\n",
    "#     params.requires_grad = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# n_features = model.classifier[0].in_features\n",
    "# n_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model.classifier = nn.Sequential(\n",
    "#     nn.Linear(n_features, 1024),\n",
    "#     nn.ReLU(),\n",
    "#     nn.Dropout(0.4),\n",
    "#     nn.Linear(1024, targets_size),\n",
    "# )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Original Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 16;\n                var nbb_unformatted_code = \"class CNN(nn.Module):\\n    def __init__(self, K):\\n        super(CNN, self).__init__()\\n        self.conv_layers = nn.Sequential(\\n            # conv1\\n            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(32),\\n            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(32),\\n            nn.MaxPool2d(2),\\n            # conv2\\n            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(64),\\n            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(64),\\n            nn.MaxPool2d(2),\\n            # conv3\\n            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(128),\\n            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(128),\\n            nn.MaxPool2d(2),\\n            # conv4\\n            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(256),\\n            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(256),\\n            nn.MaxPool2d(2),\\n        )\\n\\n        self.dense_layers = nn.Sequential(\\n            nn.Dropout(0.4),\\n            nn.Linear(50176, 1024),\\n            nn.ReLU(),\\n            nn.Dropout(0.4),\\n            nn.Linear(1024, K),\\n        )\\n\\n    def forward(self, X):\\n        out = self.conv_layers(X)\\n\\n        # Flatten\\n        out = out.view(-1, 50176)\\n\\n        # Fully connected\\n        out = self.dense_layers(out)\\n\\n        return out\";\n                var nbb_formatted_code = \"class CNN(nn.Module):\\n    def __init__(self, K):\\n        super(CNN, self).__init__()\\n        self.conv_layers = nn.Sequential(\\n            # conv1\\n            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(32),\\n            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(32),\\n            nn.MaxPool2d(2),\\n            # conv2\\n            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(64),\\n            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(64),\\n            nn.MaxPool2d(2),\\n            # conv3\\n            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(128),\\n            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(128),\\n            nn.MaxPool2d(2),\\n            # conv4\\n            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(256),\\n            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),\\n            nn.ReLU(),\\n            nn.BatchNorm2d(256),\\n            nn.MaxPool2d(2),\\n        )\\n\\n        self.dense_layers = nn.Sequential(\\n            nn.Dropout(0.4),\\n            nn.Linear(50176, 1024),\\n            nn.ReLU(),\\n            nn.Dropout(0.4),\\n            nn.Linear(1024, K),\\n        )\\n\\n    def forward(self, X):\\n        out = self.conv_layers(X)\\n\\n        # Flatten\\n        out = out.view(-1, 50176)\\n\\n        # Fully connected\\n        out = self.dense_layers(out)\\n\\n        return out\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "class CNN(nn.Module):\n",
    "    def __init__(self, K):\n",
    "        super(CNN, self).__init__()\n",
    "        self.conv_layers = nn.Sequential(\n",
    "            # conv1\n",
    "            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(32),\n",
    "            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(32),\n",
    "            nn.MaxPool2d(2),\n",
    "            # conv2\n",
    "            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(64),\n",
    "            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(64),\n",
    "            nn.MaxPool2d(2),\n",
    "            # conv3\n",
    "            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(128),\n",
    "            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(128),\n",
    "            nn.MaxPool2d(2),\n",
    "            # conv4\n",
    "            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(256),\n",
    "            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.BatchNorm2d(256),\n",
    "            nn.MaxPool2d(2),\n",
    "        )\n",
    "\n",
    "        self.dense_layers = nn.Sequential(\n",
    "            nn.Dropout(0.4),\n",
    "            nn.Linear(50176, 1024),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.4),\n",
    "            nn.Linear(1024, K),\n",
    "        )\n",
    "\n",
    "    def forward(self, X):\n",
    "        out = self.conv_layers(X)\n",
    "\n",
    "        # Flatten\n",
    "        out = out.view(-1, 50176)\n",
    "\n",
    "        # Fully connected\n",
    "        out = self.dense_layers(out)\n",
    "\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cpu\n"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 18;\n                var nbb_unformatted_code = \"device = torch.device(\\\"cuda\\\" if torch.cuda.is_available() else \\\"cpu\\\")\\nprint(device)\";\n                var nbb_formatted_code = \"device = torch.device(\\\"cuda\\\" if torch.cuda.is_available() else \\\"cpu\\\")\\nprint(device)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 19;\n                var nbb_unformatted_code = \"device = \\\"cpu\\\"\";\n                var nbb_formatted_code = \"device = \\\"cpu\\\"\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "device = \"cpu\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 20;\n                var nbb_unformatted_code = \"model = CNN(targets_size)\";\n                var nbb_formatted_code = \"model = CNN(targets_size)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model = CNN(targets_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CNN(\n",
       "  (conv_layers): Sequential(\n",
       "    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): ReLU()\n",
       "    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (4): ReLU()\n",
       "    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (8): ReLU()\n",
       "    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (11): ReLU()\n",
       "    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (15): ReLU()\n",
       "    (16): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (17): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (18): ReLU()\n",
       "    (19): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (21): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (22): ReLU()\n",
       "    (23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (24): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (25): ReLU()\n",
       "    (26): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "  )\n",
       "  (dense_layers): Sequential(\n",
       "    (0): Dropout(p=0.4, inplace=False)\n",
       "    (1): Linear(in_features=50176, out_features=1024, bias=True)\n",
       "    (2): ReLU()\n",
       "    (3): Dropout(p=0.4, inplace=False)\n",
       "    (4): Linear(in_features=1024, out_features=39, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 21;\n                var nbb_unformatted_code = \"model.to(device)\";\n                var nbb_formatted_code = \"model.to(device)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "----------------------------------------------------------------\n",
      "        Layer (type)               Output Shape         Param #\n",
      "================================================================\n",
      "            Conv2d-1         [-1, 32, 224, 224]             896\n",
      "              ReLU-2         [-1, 32, 224, 224]               0\n",
      "       BatchNorm2d-3         [-1, 32, 224, 224]              64\n",
      "            Conv2d-4         [-1, 32, 224, 224]           9,248\n",
      "              ReLU-5         [-1, 32, 224, 224]               0\n",
      "       BatchNorm2d-6         [-1, 32, 224, 224]              64\n",
      "         MaxPool2d-7         [-1, 32, 112, 112]               0\n",
      "            Conv2d-8         [-1, 64, 112, 112]          18,496\n",
      "              ReLU-9         [-1, 64, 112, 112]               0\n",
      "      BatchNorm2d-10         [-1, 64, 112, 112]             128\n",
      "           Conv2d-11         [-1, 64, 112, 112]          36,928\n",
      "             ReLU-12         [-1, 64, 112, 112]               0\n",
      "      BatchNorm2d-13         [-1, 64, 112, 112]             128\n",
      "        MaxPool2d-14           [-1, 64, 56, 56]               0\n",
      "           Conv2d-15          [-1, 128, 56, 56]          73,856\n",
      "             ReLU-16          [-1, 128, 56, 56]               0\n",
      "      BatchNorm2d-17          [-1, 128, 56, 56]             256\n",
      "           Conv2d-18          [-1, 128, 56, 56]         147,584\n",
      "             ReLU-19          [-1, 128, 56, 56]               0\n",
      "      BatchNorm2d-20          [-1, 128, 56, 56]             256\n",
      "        MaxPool2d-21          [-1, 128, 28, 28]               0\n",
      "           Conv2d-22          [-1, 256, 28, 28]         295,168\n",
      "             ReLU-23          [-1, 256, 28, 28]               0\n",
      "      BatchNorm2d-24          [-1, 256, 28, 28]             512\n",
      "           Conv2d-25          [-1, 256, 28, 28]         590,080\n",
      "             ReLU-26          [-1, 256, 28, 28]               0\n",
      "      BatchNorm2d-27          [-1, 256, 28, 28]             512\n",
      "        MaxPool2d-28          [-1, 256, 14, 14]               0\n",
      "          Dropout-29                [-1, 50176]               0\n",
      "           Linear-30                 [-1, 1024]      51,381,248\n",
      "             ReLU-31                 [-1, 1024]               0\n",
      "          Dropout-32                 [-1, 1024]               0\n",
      "           Linear-33                   [-1, 39]          39,975\n",
      "================================================================\n",
      "Total params: 52,595,399\n",
      "Trainable params: 52,595,399\n",
      "Non-trainable params: 0\n",
      "----------------------------------------------------------------\n",
      "Input size (MB): 0.57\n",
      "Forward/backward pass size (MB): 143.96\n",
      "Params size (MB): 200.64\n",
      "Estimated Total Size (MB): 345.17\n",
      "----------------------------------------------------------------\n"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 22;\n                var nbb_unformatted_code = \"from torchsummary import summary\\n\\nsummary(model, (3, 224, 224))\";\n                var nbb_formatted_code = \"from torchsummary import summary\\n\\nsummary(model, (3, 224, 224))\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from torchsummary import summary\n",
    "\n",
    "summary(model, (3, 224, 224))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 30;\n                var nbb_unformatted_code = \"criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss\\noptimizer = torch.optim.Adam(model.parameters())\";\n                var nbb_formatted_code = \"criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss\\noptimizer = torch.optim.Adam(model.parameters())\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss\n",
    "optimizer = torch.optim.Adam(model.parameters())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Batch Gradient Descent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 31;\n                var nbb_unformatted_code = \"def batch_gd(model, criterion, train_loader, test_laoder, epochs):\\n    train_losses = np.zeros(epochs)\\n    test_losses = np.zeros(epochs)\\n\\n    for e in range(epochs):\\n        t0 = datetime.now()\\n        train_loss = []\\n        for inputs, targets in train_loader:\\n            inputs, targets = inputs.to(device), targets.to(device)\\n\\n            optimizer.zero_grad()\\n\\n            output = model(inputs)\\n\\n            loss = criterion(output, targets)\\n\\n            train_loss.append(loss.item())  # torch to numpy world\\n\\n            loss.backward()\\n            optimizer.step()\\n\\n        train_loss = np.mean(train_loss)\\n\\n        validation_loss = []\\n\\n        for inputs, targets in validation_loader:\\n\\n            inputs, targets = inputs.to(device), targets.to(device)\\n\\n            output = model(inputs)\\n\\n            loss = criterion(output, targets)\\n\\n            validation_loss.append(loss.item())  # torch to numpy world\\n\\n        validation_loss = np.mean(validation_loss)\\n\\n        train_losses[e] = train_loss\\n        validation_losses[e] = validation_loss\\n\\n        dt = datetime.now() - t0\\n\\n        print(\\n            f\\\"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}\\\"\\n        )\\n\\n    return train_losses, validation_losses\";\n                var nbb_formatted_code = \"def batch_gd(model, criterion, train_loader, test_laoder, epochs):\\n    train_losses = np.zeros(epochs)\\n    test_losses = np.zeros(epochs)\\n\\n    for e in range(epochs):\\n        t0 = datetime.now()\\n        train_loss = []\\n        for inputs, targets in train_loader:\\n            inputs, targets = inputs.to(device), targets.to(device)\\n\\n            optimizer.zero_grad()\\n\\n            output = model(inputs)\\n\\n            loss = criterion(output, targets)\\n\\n            train_loss.append(loss.item())  # torch to numpy world\\n\\n            loss.backward()\\n            optimizer.step()\\n\\n        train_loss = np.mean(train_loss)\\n\\n        validation_loss = []\\n\\n        for inputs, targets in validation_loader:\\n\\n            inputs, targets = inputs.to(device), targets.to(device)\\n\\n            output = model(inputs)\\n\\n            loss = criterion(output, targets)\\n\\n            validation_loss.append(loss.item())  # torch to numpy world\\n\\n        validation_loss = np.mean(validation_loss)\\n\\n        train_losses[e] = train_loss\\n        validation_losses[e] = validation_loss\\n\\n        dt = datetime.now() - t0\\n\\n        print(\\n            f\\\"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}\\\"\\n        )\\n\\n    return train_losses, validation_losses\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def batch_gd(model, criterion, train_loader, test_laoder, epochs):\n",
    "    train_losses = np.zeros(epochs)\n",
    "    validation_losses = np.zeros(epochs)\n",
    "\n",
    "    for e in range(epochs):\n",
    "        t0 = datetime.now()\n",
    "        train_loss = []\n",
    "        for inputs, targets in train_loader:\n",
    "            inputs, targets = inputs.to(device), targets.to(device)\n",
    "\n",
    "            optimizer.zero_grad()\n",
    "\n",
    "            output = model(inputs)\n",
    "\n",
    "            loss = criterion(output, targets)\n",
    "\n",
    "            train_loss.append(loss.item())  # torch to numpy world\n",
    "\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "\n",
    "        train_loss = np.mean(train_loss)\n",
    "\n",
    "        validation_loss = []\n",
    "\n",
    "        for inputs, targets in validation_loader:\n",
    "\n",
    "            inputs, targets = inputs.to(device), targets.to(device)\n",
    "\n",
    "            output = model(inputs)\n",
    "\n",
    "            loss = criterion(output, targets)\n",
    "\n",
    "            validation_loss.append(loss.item())  # torch to numpy world\n",
    "\n",
    "        validation_loss = np.mean(validation_loss)\n",
    "\n",
    "        train_losses[e] = train_loss\n",
    "        validation_losses[e] = validation_loss\n",
    "\n",
    "        dt = datetime.now() - t0\n",
    "\n",
    "        print(\n",
    "            f\"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}\"\n",
    "        )\n",
    "\n",
    "    return train_losses, validation_losses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 32;\n                var nbb_unformatted_code = \"device = \\\"cpu\\\"\";\n                var nbb_formatted_code = \"device = \\\"cpu\\\"\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "device = \"cpu\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 33;\n                var nbb_unformatted_code = \"batch_size = 64\\ntrain_loader = torch.utils.data.DataLoader(\\n    dataset, batch_size=batch_size, sampler=train_sampler\\n)\\ntest_loader = torch.utils.data.DataLoader(\\n    dataset, batch_size=batch_size, sampler=test_sampler\\n)\\nvalidation_loader = torch.utils.data.DataLoader(\\n    dataset, batch_size=batch_size, sampler=validation_sampler\\n)\";\n                var nbb_formatted_code = \"batch_size = 64\\ntrain_loader = torch.utils.data.DataLoader(\\n    dataset, batch_size=batch_size, sampler=train_sampler\\n)\\ntest_loader = torch.utils.data.DataLoader(\\n    dataset, batch_size=batch_size, sampler=test_sampler\\n)\\nvalidation_loader = torch.utils.data.DataLoader(\\n    dataset, batch_size=batch_size, sampler=validation_sampler\\n)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "batch_size = 64\n",
    "train_loader = torch.utils.data.DataLoader(\n",
    "    dataset, batch_size=batch_size, sampler=train_sampler\n",
    ")\n",
    "test_loader = torch.utils.data.DataLoader(\n",
    "    dataset, batch_size=batch_size, sampler=test_sampler\n",
    ")\n",
    "validation_loader = torch.utils.data.DataLoader(\n",
    "    dataset, batch_size=batch_size, sampler=validation_sampler\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 32;\n                var nbb_unformatted_code = \"# train_losses, validation_losses = batch_gd(\\n#     model, criterion, train_loader, validation_loader, 5\\n# )\";\n                var nbb_formatted_code = \"# train_losses, validation_losses = batch_gd(\\n#     model, criterion, train_loader, validation_loader, 5\\n# )\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_losses, validation_losses = batch_gd(\n",
    "    model, criterion, train_loader, validation_loader, 5\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Save the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 34;\n                var nbb_unformatted_code = \"# torch.save(model.state_dict() , 'plant_disease_model_1.pt')\";\n                var nbb_formatted_code = \"# torch.save(model.state_dict() , 'plant_disease_model_1.pt')\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# torch.save(model.state_dict() , 'plant_disease_model_1.pt')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CNN(\n",
       "  (conv_layers): Sequential(\n",
       "    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): ReLU()\n",
       "    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (4): ReLU()\n",
       "    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (8): ReLU()\n",
       "    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (11): ReLU()\n",
       "    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (15): ReLU()\n",
       "    (16): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (17): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (18): ReLU()\n",
       "    (19): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (21): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (22): ReLU()\n",
       "    (23): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (24): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (25): ReLU()\n",
       "    (26): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "  )\n",
       "  (dense_layers): Sequential(\n",
       "    (0): Dropout(p=0.4, inplace=False)\n",
       "    (1): Linear(in_features=50176, out_features=1024, bias=True)\n",
       "    (2): ReLU()\n",
       "    (3): Dropout(p=0.4, inplace=False)\n",
       "    (4): Linear(in_features=1024, out_features=39, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "targets_size = 39\n",
    "model = CNN(targets_size)\n",
    "model.load_state_dict(torch.load(\"plant_disease_model_1_latest.pt\"))\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %matplotlib notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Plot the loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(train_losses , label = 'train_loss')\n",
    "plt.plot(validation_losses , label = 'validation_loss')\n",
    "plt.xlabel('No of Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 35;\n                var nbb_unformatted_code = \"def accuracy(loader):\\n    n_correct = 0\\n    n_total = 0\\n\\n    for inputs , targets in loader:\\n        inputs , targets = inputs.to(device) , targets.to(device)\\n\\n        outputs = model(inputs)\\n\\n        _ , predictions = torch.max(outputs,1)\\n\\n        n_correct += (predictions == targets).sum().item()\\n        n_total += targets.shape[0]\\n\\n\\n    acc = n_correct / n_total\\n    return acc\";\n                var nbb_formatted_code = \"def accuracy(loader):\\n    n_correct = 0\\n    n_total = 0\\n\\n    for inputs, targets in loader:\\n        inputs, targets = inputs.to(device), targets.to(device)\\n\\n        outputs = model(inputs)\\n\\n        _, predictions = torch.max(outputs, 1)\\n\\n        n_correct += (predictions == targets).sum().item()\\n        n_total += targets.shape[0]\\n\\n    acc = n_correct / n_total\\n    return acc\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def accuracy(loader):\n",
    "    n_correct = 0\n",
    "    n_total = 0\n",
    "\n",
    "    for inputs, targets in loader:\n",
    "        inputs, targets = inputs.to(device), targets.to(device)\n",
    "\n",
    "        outputs = model(inputs)\n",
    "\n",
    "        _, predictions = torch.max(outputs, 1)\n",
    "\n",
    "        n_correct += (predictions == targets).sum().item()\n",
    "        n_total += targets.shape[0]\n",
    "\n",
    "    acc = n_correct / n_total\n",
    "    return acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_acc = accuracy(train_loader)\n",
    "test_acc = accuracy(test_loader)\n",
    "validation_acc = accuracy(validation_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train Accuracy : 96.7\n",
      "Test Accuracy : 98.9\n",
      "Validation Accuracy : 98.7\n"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 38;\n                var nbb_unformatted_code = \"print(f\\\"Train Accuracy : {train_acc}\\\\nTest Accuracy : {test_acc}\\\\nValidation Accuracy : {validation_acc}\\\")\";\n                var nbb_formatted_code = \"print(\\n    f\\\"Train Accuracy : {train_acc}\\\\nTest Accuracy : {test_acc}\\\\nValidation Accuracy : {validation_acc}\\\"\\n)\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\n",
    "    f\"Train Accuracy : {train_acc}\\nTest Accuracy : {test_acc}\\nValidation Accuracy : {validation_acc}\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Single Image Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'dataset' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-9-0e3bd74576a2>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtransform_index_to_disease\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdataset\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mclass_to_idx\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'dataset' is not defined"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 9;\n                var nbb_unformatted_code = \"transform_index_to_disease = dataset.class_to_idx\";\n                var nbb_formatted_code = \"transform_index_to_disease = dataset.class_to_idx\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "transform_index_to_disease = dataset.class_to_idx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'transform_index_to_disease' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-10-1fe109ff4fe8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m transform_index_to_disease = dict(\n\u001b[1;32m----> 2\u001b[1;33m     \u001b[1;33m[\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalue\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtransform_index_to_disease\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m )  # reverse the index\n",
      "\u001b[1;31mNameError\u001b[0m: name 'transform_index_to_disease' is not defined"
     ]
    },
    {
     "data": {
      "application/javascript": "\n            setTimeout(function() {\n                var nbb_cell_id = 10;\n                var nbb_unformatted_code = \"transform_index_to_disease = dict(\\n    [(value, key) for key, value in transform_index_to_disease.items()]\\n)  # reverse the index\";\n                var nbb_formatted_code = \"transform_index_to_disease = dict(\\n    [(value, key) for key, value in transform_index_to_disease.items()]\\n)  # reverse the index\";\n                var nbb_cells = Jupyter.notebook.get_cells();\n                for (var i = 0; i < nbb_cells.length; ++i) {\n                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n                             nbb_cells[i].set_text(nbb_formatted_code);\n                        }\n                        break;\n                    }\n                }\n            }, 500);\n            ",
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "transform_index_to_disease = dict(\n",
    "    [(value, key) for key, value in transform_index_to_disease.items()]\n",
    ")  # reverse the index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"disease_info.csv\", encoding=\"cp1252\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "import torchvision.transforms.functional as TF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def single_prediction(image_path):\n",
    "    image = Image.open(image_path)\n",
    "    image = image.resize((224, 224))\n",
    "    input_data = TF.to_tensor(image)\n",
    "    input_data = input_data.view((-1, 3, 224, 224))\n",
    "    output = model(input_data)\n",
    "    output = output.detach().numpy()\n",
    "    index = np.argmax(output)\n",
    "    print(\"Original : \", image_path[12:-4])\n",
    "    pred_csv = data[\"disease_name\"][index]\n",
    "    print(pred_csv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  Apple_ceder_apple_rust\n",
      "Apple : Cedar rust\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/Apple_ceder_apple_rust.JPG\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Wrong Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  Apple_scab\n",
      "Tomato : Septoria Leaf Spot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/Apple_scab.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  Grape_esca\n",
      "Grape : Esca | Black Measles\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/Grape_esca.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  apple_black_rot\n",
      "Pepper bell : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/apple_black_rot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  apple_healthy\n",
      "Apple : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/apple_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  background_without_leaves\n",
      "Background Without Leaves\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/background_without_leaves.jpg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  blueberry_healthy\n",
      "Blueberry : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/blueberry_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  cherry_healthy\n",
      "Cherry : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/cherry_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  cherry_powdery_mildew\n",
      "Cherry : Powdery Mildew\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/cherry_powdery_mildew.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  corn_cercospora_leaf\n",
      "Corn : Cercospora Leaf Spot | Gray Leaf Spot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/corn_cercospora_leaf.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  corn_common_rust\n",
      "Corn : Common Rust\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/corn_common_rust.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  corn_healthy\n",
      "Corn : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/corn_healthy.jpg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  corn_northen_leaf_blight\n",
      "Corn : Northern Leaf Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/corn_northen_leaf_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  grape_black_rot\n",
      "Grape : Black Rot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/grape_black_rot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  grape_healthy\n",
      "Grape : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/grape_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  grape_leaf_blight\n",
      "Grape : Leaf Blight | Isariopsis Leaf Spot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/grape_leaf_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  orange_haunglongbing\n",
      "Orange : Haunglongbing | Citrus Greening\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/orange_haunglongbing.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  peach_bacterial_spot\n",
      "Peach : Bacterial Spot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/peach_bacterial_spot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  peach_healthy\n",
      "Peach : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/peach_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  pepper_bacterial_spot\n",
      "Pepper bell : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/pepper_bacterial_spot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  pepper_bell_healthy\n",
      "Pepper bell : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/pepper_bell_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  potato_early_blight\n",
      "Potato : Early Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/potato_early_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  potato_healthy\n",
      "Potato : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/potato_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  potato_late_blight\n",
      "Potato : Late Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/potato_late_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  raspberry_healthy\n",
      "Raspberry : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/raspberry_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  soyaben healthy\n",
      "Soybean : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/soyaben healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  potato_late_blight\n",
      "Potato : Late Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/potato_late_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  squash_powdery_mildew\n",
      "Squash : Powdery Mildew\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/squash_powdery_mildew.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  starwberry_healthy\n",
      "Strawberry : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/starwberry_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  starwberry_leaf_scorch\n",
      "Strawberry : Leaf Scorch\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/starwberry_leaf_scorch.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_bacterial_spot\n",
      "Tomato : Early Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_bacterial_spot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_early_blight\n",
      "Tomato : Early Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_early_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_healthy\n",
      "Tomato : Healthy\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_healthy.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_late_blight\n",
      "Tomato : Late Blight\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_late_blight.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_leaf_mold\n",
      "Tomato : Leaf Mold\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_leaf_mold.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_mosaic_virus\n",
      "Tomato : Mosaic Virus\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_mosaic_virus.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_septoria_leaf_spot\n",
      "Tomato : Septoria Leaf Spot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_septoria_leaf_spot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_spider_mites_two_spotted_spider_mites\n",
      "Tomato : Spider Mites | Two-Spotted Spider Mite\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_spider_mites_two_spotted_spider_mites.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_target_spot\n",
      "Tomato : Target Spot\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_target_spot.JPG\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original :  tomato_yellow_leaf_curl_virus\n",
      "Tomato : Yellow Leaf Curl Virus\n"
     ]
    }
   ],
   "source": [
    "single_prediction(\"test_images/tomato_yellow_leaf_curl_virus.JPG\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
