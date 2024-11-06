from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message')
    
    # For simplicity, we're sending a static response.
    # You can replace this with any logic, such as querying a database or calling a chatbot API.
    bot_reply = "You said: " + user_message  # Simple echo for demonstration
    
    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)