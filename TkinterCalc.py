import tkinter
import tkinter.messagebox
import torch
import torch.nn.functional as F
import math
import json
import os
from datetime import datetime

#Button Layout
button_values = [
    ["AC", "+/-", "%", "÷"], 
    ["7", "8", "9", "×"], 
    ["4", "5", "6", "-"],
    ["1", "2", "3", "+"],
    ["0", ".", "√", "="]
]

right_symbols = ["÷", "×", "-", "+", "="]
top_symbols = ["AC", "+/-", "%"]

row_count = len(button_values)
column_count = len(button_values[0])

#Colours
color_light_grey = "White"
color_black = "#9966CC"
color_dark_grey = "Pink"
color_orange = "Light Blue"
color_white = "White"

#Window setup
window = tkinter.Tk()
window.title("AI Powered Calculator by Hiba")
window.resizable(False, False)
window.configure(bg="black")
window.minsize(400, 600)

#Frame setup
frame = tkinter.Frame(window)
frame.pack(fill=tkinter.BOTH, expand=True)

#Prediction label for AI suggestion
prediction_label = tkinter.Label(frame, text="", font=("Arial", 20), background=color_black, foreground="White", anchor="e")
prediction_label.grid(row=0, column=0, columnspan=column_count, sticky="nsew", padx=1, pady=1)

#Main display label
label = tkinter.Label(frame, text="0", font=("Arial", 45), background=color_black, 
                      foreground=color_white, anchor="e", width=12)
label.grid(row=1, column=0, columnspan=column_count, sticky="nsew", padx=1, pady=1)

#Calculator state
A = "0"
operator = None
B = None
calculation_history = []
history_numbers = [] #track last 10 numbers
history_operators = [] #track last 10 operators

# PyTorch-based AI features
ai_prediction_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)  # Weighted prediction
ai_confidence_threshold = 0.7
ai_pattern_memory = []

# Load history from file
def load_history():
    global calculation_history
    try:
        if os.path.exists("calc_history.json"):
            with open("calc_history.json", "r") as f:
                calculation_history = json.load(f)
    except:
        calculation_history = []

# Save history to file
def save_history():
    try:
        with open("calc_history.json", "w") as f:
            json.dump(calculation_history[-50:], f)  # Keep last 50 calculations
    except:
        pass

def clear_all():
    global A, B, operator
    A = "0"
    operator = None
    B = None

def remove_zero_decimal(num):
    if num % 1 == 0:
        num = int(num)
    return str(num)

def format_display_number(num):
    """Format number for display with proper length and precision"""
    if isinstance(num, str):
        try:
            num = float(num)
        except:
            return num
    
    # Handle very large or very small numbers with scientific notation
    if abs(num) >= 1e10 or (abs(num) < 1e-6 and num != 0):
        return f"{num:.3e}"
    
    # For normal numbers, limit to reasonable precision
    if num % 1 == 0:  # Integer
        result = str(int(num))
    else:  # Float
        # Limit decimal places based on magnitude
        if abs(num) >= 1000:
            result = f"{num:.2f}".rstrip('0').rstrip('.')
        elif abs(num) >= 1:
            result = f"{num:.6f}".rstrip('0').rstrip('.')
        else:
            result = f"{num:.8f}".rstrip('0').rstrip('.')
    
    # Truncate if too long for display (keep first 10 characters)
    if len(result) > 10:
        if 'e' in result.lower():
            return result[:10]
        else:
            return result[:10]
    
    return result

def show_status(message):
    """Show temporary status message in prediction label"""
    prediction_label["text"] = message
    window.after(2000, lambda: prediction_label.config(text=""))

def save_calculation(expression, result):
    """Save calculation to history"""
    global calculation_history
    calculation_history.append({
        "expression": expression,
        "result": result,
        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    })
    save_history()

def show_history():
    """Show calculation history in a popup"""
    if not calculation_history:
        tkinter.messagebox.showinfo("History", "No calculations yet!")
        return
    
    history_text = "Recent Calculations:\n\n"
    for calc in calculation_history[-10:]:  # Show last 10
        history_text += f"{calc['expression']} = {calc['result']}\n"
    
    tkinter.messagebox.showinfo("Calculation History", history_text)

#AI Prediction with Enhanced PyTorch
def predict_next():
    global ai_confidence_threshold, ai_pattern_memory
    prediction_text = ""
    
    # Advanced PyTorch-based predictions
    if len(history_numbers) >= 4:
        try:
            # Create tensor from recent numbers
            recent_tensor = torch.tensor(history_numbers[-4:], dtype=torch.float32)
            
            # Weighted prediction using PyTorch operations
            weighted_pred = torch.dot(recent_tensor, ai_prediction_weights).item()
            
            # Pattern analysis using tensor operations
            differences = torch.diff(recent_tensor)
            pattern_score = calculate_pattern_strength(differences)
            
            # Trend analysis
            trend_direction = "↗" if differences[-1] > 0 else "↘" if differences[-1] < 0 else "→"
            
            prediction_text += f"AI: {format_display_number(weighted_pred)} {trend_direction}"
            
            # Add confidence indicator based on pattern strength
            if pattern_score > ai_confidence_threshold:
                prediction_text += " (Strong Pattern)"
            elif pattern_score > 0.4:
                prediction_text += " (Pattern Detected)"
            
        except Exception:
            # Fallback to simple prediction
            simple_prediction()
            return
            
    elif len(history_numbers) >= 2:
        simple_prediction()
        return
    
    # Enhanced operator prediction with PyTorch
    if history_operators:
        op_prediction = predict_operator_with_pytorch()
        prediction_text += f" | Next Op: {op_prediction}"
    
    prediction_label["text"] = prediction_text

def simple_prediction():
    """Simple PyTorch prediction for fewer data points"""
    if len(history_numbers) >= 2:
        tensor = torch.tensor(history_numbers[-2:], dtype=torch.float32)
        predicted_num = tensor.mean().item()
        prediction_label["text"] = f"Learning: {format_display_number(predicted_num)}"

def calculate_pattern_strength(differences):
    """Calculate pattern strength using PyTorch operations"""
    if len(differences) < 2:
        return 0.0
    
    # Check for consistent differences (arithmetic sequence)
    diff_variance = torch.var(differences).item()
    
    # Check for consistent ratios (geometric-like patterns)
    abs_diffs = torch.abs(differences)
    ratio_consistency = 1.0 / (1.0 + torch.std(abs_diffs).item())
    
    # Combine scores
    pattern_strength = max(0.0, min(1.0, ratio_consistency - diff_variance * 0.1))
    
    # Store pattern for learning
    ai_pattern_memory.append(pattern_strength)
    if len(ai_pattern_memory) > 20:
        ai_pattern_memory.pop(0)
    
    return pattern_strength

def predict_operator_with_pytorch():
    """Enhanced operator prediction using PyTorch"""
    if not history_operators:
        return "+"
    
    # Convert operators to numerical values for tensor operations
    op_to_num = {"+": 1, "-": 2, "×": 3, "÷": 4}
    num_to_op = {1: "+", 2: "-", 3: "×", 4: "÷"}
    
    # Create tensor from recent operator history
    recent_ops = history_operators[-min(6, len(history_operators)):]
    op_tensor = torch.tensor([op_to_num.get(op, 1) for op in recent_ops], dtype=torch.float32)
    
    # Find most frequent using PyTorch operations
    unique_ops = torch.unique(op_tensor)
    frequencies = torch.zeros_like(unique_ops)
    
    for i, unique_op in enumerate(unique_ops):
        frequencies[i] = torch.sum(op_tensor == unique_op)
    
    # Get most frequent operator
    most_frequent_idx = torch.argmax(frequencies)
    predicted_op_num = unique_ops[most_frequent_idx].item()
    
    return num_to_op.get(int(predicted_op_num), "+")

def analyze_calculation_patterns():
    """Analyze patterns in calculation history using PyTorch"""
    if len(calculation_history) < 3:
        return "Insufficient data for analysis"
    
    # Extract results from history
    results = []
    for calc in calculation_history[-10:]:  # Last 10 calculations
        try:
            result = float(calc['result'])
            results.append(result)
        except:
            continue
    
    if len(results) < 3:
        return "No valid numerical results"
    
    # PyTorch analysis
    results_tensor = torch.tensor(results, dtype=torch.float32)
    
    # Statistical analysis
    mean_val = torch.mean(results_tensor).item()
    std_val = torch.std(results_tensor).item()
    median_val = torch.median(results_tensor).item()
    
    # Trend analysis
    if len(results) >= 3:
        recent_trend = torch.mean(results_tensor[-3:]).item() - torch.mean(results_tensor[:-3]).item()
        trend_text = "increasing" if recent_trend > 0 else "decreasing" if recent_trend < 0 else "stable"
    else:
        trend_text = "unknown"
    
    analysis = f"""PyTorch Pattern Analysis:

Recent Results Trend: {trend_text}
Average Result: {mean_val:.2f}
Median Result: {median_val:.2f}
Result Variance: {std_val:.2f}

Pattern Memory: {len(ai_pattern_memory)} samples
Avg Pattern Strength: {sum(ai_pattern_memory)/len(ai_pattern_memory):.2f if ai_pattern_memory else 0:.2f}"""
    
    return analysis

def update_ai_learning(result_value):
    """Update AI learning with new calculation result"""
    global ai_prediction_weights
    
    try:
        # Adaptive weight adjustment based on prediction accuracy
        if len(history_numbers) >= 4:
            recent_tensor = torch.tensor(history_numbers[-4:], dtype=torch.float32)
            predicted = torch.dot(recent_tensor, ai_prediction_weights).item()
            actual = float(result_value)
            
            # Calculate prediction error
            error = abs(predicted - actual)
            
            # Adjust weights slightly based on error (simple learning)
            if error > 10:  # Large error, adjust weights more
                adjustment = torch.tensor([0.01, 0.01, -0.01, -0.01], dtype=torch.float32)
            else:  # Small error, minimal adjustment
                adjustment = torch.tensor([0.005, 0.005, -0.005, -0.005], dtype=torch.float32)
            
            # Apply adjustment and normalize
            ai_prediction_weights += adjustment
            ai_prediction_weights = F.normalize(ai_prediction_weights, p=1, dim=0)
            
    except Exception:
        pass  # Fail silently if learning update fails

def clear_all():
    global A, B, operator
    A = "0"
    operator = None
    B = None

#Button click handler
def button_clicked(value):
    global right_symbols, label, A, B, operator, history_numbers, history_operators

    if value in top_symbols:
        if value == "AC":
            clear_all()
            label["text"] = "0"
            prediction_label["text"] = ""
        elif value == "+/-":
            result = float(label["text"]) * -1
            label["text"] = format_display_number(result)
        elif value == "%":
            result = float(label["text"]) / 100
            label["text"] = format_display_number(result)

    elif value in right_symbols:
        if value == "=":
            if A is not None and operator is not None:
                B = label["text"]
                numA = float(A)
                numB = float(B) 
                
                # Create expression string for history
                expression = f"{A} {operator} {B}"

                if operator == "+":
                    result = numA + numB
                    label["text"] = format_display_number(result)
                elif operator == "-":
                    result = numA - numB
                    label["text"] = format_display_number(result)      
                elif operator == "×":
                    result = numA * numB
                    label["text"] = format_display_number(result)
                elif operator == "÷":
                    if numB != 0:
                        result = numA / numB
                        label["text"] = format_display_number(result)
                    else:
                        label["text"] = "Error"
                        clear_all()
                        return

                # Save to history and update AI learning
                save_calculation(expression, label["text"])
                update_ai_learning(label["text"])

                #Save history
                history_numbers.append(float(label["text"]))
                if len(history_numbers) > 10:
                    history_numbers.pop(0)
                history_operators.append(operator)
                if len(history_operators) > 10:
                    history_operators.pop(0)

                clear_all()
                predict_next()
                
        elif value in "+-×÷":
            if operator is None:
                A = label["text"]
                label["text"] = "0"
                B = "0"
            operator = value  

    elif value in top_symbols:
        if value == "AC":
            clear_all()
            label["text"] = "0"
            prediction_label["text"] = ""
        elif value == "+/-":
            result = float(label["text"]) * -1
            label["text"] = format_display_number(result)
        elif value == "%":
            result = float(label["text"]) / 100
            label["text"] = format_display_number(result)

    else: #digits, . or √
        if value == ".":
            if value not in label["text"] and len(label["text"]) < 10:
                label["text"] += value
        elif value == "√":
            try:
                import math
                result = math.sqrt(float(label["text"]))
                label["text"] = format_display_number(result)
            except ValueError:
                label["text"] = "Error"
        elif value in "0123456789":
            if label["text"] == "0":
                label["text"] = value
            else: 
                # Limit input length to prevent layout breaking
                if len(label["text"]) < 10:
                    label["text"] += value

#Create buttons
for row in range(len(button_values)):
    for column in range(len(button_values[row])):
        value = button_values[row][column]
        button = tkinter.Button(frame, text=value, font=("Arial", 30), 
                               command=lambda value=value: button_clicked(value),
                               bd=0, relief="flat"
                                )
        
        if value in top_symbols:
            button.config(foreground=color_black, background=color_light_grey)
        elif value in right_symbols:
            button.config(foreground=color_white, background=color_orange)
        else:
            button.config(foreground=color_white, background=color_dark_grey)
        button.grid(row=row+2, column=column, sticky="nsew", padx=1, pady=1)

# Configure grid weights to make buttons expand
for i in range(len(button_values) + 2):  # +2 for prediction and display rows
    frame.grid_rowconfigure(i, weight=1)
for i in range(column_count):
    frame.grid_columnconfigure(i, weight=1)

# Keyboard support
def on_key_press(event):
    key = event.char
    if key.isdigit() or key == '.':
        button_clicked(key)
    elif key in '+-*/':
        if key == '*': key = '×'
        elif key == '/': key = '÷'
        button_clicked(key)
    elif key == '\r' or key == '=':  # Enter or equals
        button_clicked('=')
    elif key == '\x08':  # Backspace
        current = label["text"]
        if len(current) > 1:
            label["text"] = current[:-1]
        else:
            label["text"] = "0"
    elif key.lower() == 'c':  # Clear
        button_clicked('MC')

window.bind('<KeyPress>', on_key_press)
window.focus_set()

# Right-click context menu
def show_context_menu(event):
    context_menu = tkinter.Menu(window, tearoff=0)
    context_menu.add_command(label="Copy Result", command=copy_result)
    context_menu.add_command(label="Show History", command=show_history)
    context_menu.add_separator()
    context_menu.add_command(label="PyTorch Analysis", command=show_pytorch_analysis)
    context_menu.add_command(label="AI Learning Status", command=show_ai_learning_status)
    context_menu.add_separator()
    context_menu.add_command(label="About", command=show_about)
    context_menu.tk_popup(event.x_root, event.y_root)

def copy_result():
    window.clipboard_clear()
    window.clipboard_append(label["text"])
    show_status("Result copied to clipboard!")

def show_pytorch_analysis():
    """Show PyTorch-based pattern analysis"""
    analysis = analyze_calculation_patterns()
    tkinter.messagebox.showinfo("PyTorch Analysis", analysis)

def show_ai_learning_status():
    """Show AI learning and adaptation status"""
    global ai_prediction_weights, ai_pattern_memory
    
    status_text = f"""AI Learning Status:

PyTorch Prediction Weights:
Weight 1: {ai_prediction_weights[0]:.3f}
Weight 2: {ai_prediction_weights[1]:.3f}
Weight 3: {ai_prediction_weights[2]:.3f}
Weight 4: {ai_prediction_weights[3]:.3f}

Pattern Memory: {len(ai_pattern_memory)} samples
Recent Patterns: {ai_pattern_memory[-5:] if ai_pattern_memory else 'None'}

Learning Features:
• Adaptive weight adjustment
• Pattern strength analysis
• Trend direction detection
• Operator frequency learning

The AI adapts its predictions based on your calculation patterns!"""
    
    tkinter.messagebox.showinfo("AI Learning Status", status_text)

def show_about():
    tkinter.messagebox.showinfo("About", 
        """AI-POWERED CALCULATOR
Made by Hiba with PyTorch Machine Learning

PYTORCH AI FEATURES:
• Weighted tensor predictions
• Pattern strength analysis
• Adaptive weight learning
• Trend direction detection
• Operator frequency analysis
• Statistical pattern recognition

STANDARD FEATURES:
• Basic math operations (+, -, ×, ÷)
• Square root and percentage
• Full keyboard support
• Copy/paste functionality
• Right-click context menus

PyTorch AI learns and adapts from your calculations!""")

label.bind("<Button-3>", show_context_menu)  # Right-click on display

#Center window
window.update()
window_width = window.winfo_width() 
window_height = window.winfo_height()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window_x = int((screen_width/2) - (window_width/2))
window_y = int((screen_height/2) - (window_height/2))
window.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")

#Run calculator
load_history()  # Load saved history on startup
window.mainloop()
