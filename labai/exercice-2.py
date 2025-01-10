from itertools import product

def combine_terms(term1, term2):
    """Combine two terms if possible and return the combined term or None."""
    combined = ''
    difference = 0
    for c1, c2 in zip(term1, term2):
        if c1 != c2:
            combined += '-'
            difference += 1
        else:
            combined += c1
    return combined if difference == 1 else None

def quine_mccluskey(truth_table):
    """Simplify a Boolean truth table using the Quine-McCluskey algorithm."""
    terms = {k: v for k, v in truth_table.items() if v == 1}
    prime_implicants = set()

    while terms:
        next_terms = {}
        marked = set()
        for t1, t2 in product(terms, repeat=2):
            if t1 != t2:
                combined = combine_terms(t1, t2)
                if combined:
                    marked.update([t1, t2])
                    next_terms[combined] = 1
        prime_implicants.update(terms.keys() - marked)
        terms = next_terms

    # Filter the prime implicants to avoid unnecessary repetitions
    essential_prime_implicants = set()
    for term in prime_implicants:
        if all(term not in other_term for other_term in prime_implicants if other_term != term):
            essential_prime_implicants.add(term)

    return essential_prime_implicants

def to_full_expression(term):
    """Convert a Boolean term to a logical expression using AND and XOR."""
    variables = ['x0', 'x1', 'x2']
    expression = []
    
    for i, c in enumerate(term):
        if c == '1':
            expression.append(variables[i])  # Variable is present (x)
        elif c == '0':
            expression.append(f'({variables[i]} XOR 1)')  # Variable is negated (X XOR 1)

    return ' AND '.join(expression)

def to_minimized_expression(term):
    """Convert a minimized Boolean term to a logical expression."""
    variables = ['x0', 'x1', 'x2']
    expression = []
    
    for i, c in enumerate(term):
        if c == '1':
            expression.append(variables[i])  # Variable is present (x)
        elif c == '0':
            expression.append(f'({variables[i]} XOR 1)')  # Variable is negated (X XOR 1)
        # If it's '-', we skip adding the variable, as it's a "don't care" condition
    
    return ' AND '.join(expression)

# Example truth table for a function with 3 variables
truth_table = {
    '000': 0,
    '001': 1,
    '010': 0,
    '011': 0,
    '100': 1,
    '101': 1,
    '110': 0,
    '111': 0,
}

# Full Boolean expression (before minimization)
full_expression_terms = [term for term, value in truth_table.items() if value == 1]
full_expression = ' OR '.join(to_full_expression(term) for term in full_expression_terms)
print("Full Boolean expression:", full_expression)

# Minimized Boolean expression (after applying Quine-McCluskey)
minimal_terms = quine_mccluskey(truth_table)
minimal_expression = ' OR '.join(to_minimized_expression(term) for term in minimal_terms)
print("Minimized Boolean expression:", minimal_expression)