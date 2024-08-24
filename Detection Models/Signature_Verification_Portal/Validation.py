import re
import unicodedata
def validate_email(email):
    # Use regex to validate email format
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(pattern, email):
        return True
    return False

def validate_password(password):
    # Check password constraints
    categories = 0
    if re.search(r'[A-Z]', password):
        categories += 1
    if re.search(r'[a-z]', password):
        categories += 1
    if re.search(r'\d', password):
        categories += 1
    if re.search(r'[^A-Za-z\d]', password):
        categories += 1
    if any(unicodedata.category(c).startswith('L') and unicodedata.category(c) != 'Lu' and unicodedata.category(c) != 'Ll' for c in password):
        categories += 1
    return categories >= 3