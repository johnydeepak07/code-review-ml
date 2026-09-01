# api/main.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from analyzer.ast_features import extract_features
from analyzer.complexity import analyze_complexity
from analyzer.suggestion_generator import generate_suggestions

app = FastAPI(
    title='Code Review API',
    description='ML-powered code review using AST feature extraction',
    version='1.0.0'
)

# CORS middleware — allows the VS Code extension and any frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load model files once when the server starts — not on every request
# If these files don't exist the server will crash here with FileNotFoundError
model = joblib.load('model/readability_model.pkl')
feature_names = joblib.load('model/feature_names.pkl')


class CodeInput(BaseModel):
    code: str
    filename: str = 'snippet.py'


class ReviewResponse(BaseModel):
    readability_score: float
    grade: str
    cyclomatic_complexity: float
    max_nesting_depth: int
    naming_entropy: float
    avg_function_length: float
    has_docstrings: bool
    num_magic_numbers: int
    suggestions: str
    filename: str


@app.post('/review', response_model=ReviewResponse)
def review_code(input: CodeInput):
    # Guard: reject empty input immediately
    if not input.code.strip():
        raise HTTPException(status_code=400, detail='Code input is empty.')

    # Step 1: Parse the code into an AST and extract features
    # If the code has a syntax error, ast.parse raises SyntaxError
    try:
        features = extract_features(input.code)
    except SyntaxError as e:
        raise HTTPException(status_code=422, detail=f'Syntax error in code: {str(e)}')

    # Step 2: Run radon complexity analysis on the same code
    complexity = analyze_complexity(input.code)

    # Step 3: Build the feature dictionary the model expects
    feat_dict = {
        'cyclomatic_complexity': complexity.cyclomatic_complexity,
        'max_nesting_depth':     features.max_nesting_depth,
        'naming_entropy':        features.naming_entropy,
        'avg_function_length':   features.avg_function_length,
        'has_docstrings':        int(features.has_docstrings),
        'num_magic_numbers':     features.num_magic_numbers,
        'num_try_except':        features.num_try_except,
    }

    # Step 4: Run the XGBoost classifier
    # predict_proba returns [[prob_class_0, prob_class_1]]
    # [0][1] gives us the probability of being "readable"
    X = pd.DataFrame([feat_dict])[feature_names]
    readability_score = float(model.predict_proba(X)[0][1])

    # Step 5: Generate LLM suggestions — only calls OpenAI if issues were found
    suggestions = generate_suggestions(
        input.code, feat_dict, vars(complexity), readability_score
    )

    return ReviewResponse(
        readability_score=round(readability_score, 3),
        grade=complexity.grade,
        cyclomatic_complexity=complexity.cyclomatic_complexity,
        max_nesting_depth=features.max_nesting_depth,
        naming_entropy=features.naming_entropy,
        avg_function_length=features.avg_function_length,
        has_docstrings=features.has_docstrings,
        num_magic_numbers=features.num_magic_numbers,
        suggestions=suggestions,
        filename=input.filename
    )


@app.get('/health')
def health():
    return {'status': 'healthy', 'model': 'readability_classifier_v1'}