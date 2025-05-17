# Design Pattern Recommender System 

A web application that recommends software design patterns based on problem descriptions using SBERT with reranking.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tahsin007/Design-Pattern-Recommendation-System.git
   cd Design-Pattern-Recommendation-System
   ```
### Install The Dependencies

```bash
    pip install -r requirements.txt
```
### Running the Application
  ```bash
    python app.py
```

### Project Structure
```tree
design-pattern-recommender/
├── app.py                # Flask backend
├── model_utils.py        # Model loading and prediction functions
├── templates/
│   ├── index.html        # Main page
│   └── results.html      # Results page
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── script.js     # Client-side JavaScript
├── requirements.txt      # Python dependencies
└── README.md            # Setup instructions
```
