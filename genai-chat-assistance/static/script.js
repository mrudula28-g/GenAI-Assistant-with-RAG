async function sendMessage() {

    let input = document.getElementById("user-input")
    let message = input.value

    if (!message) return

    let messages = document.getElementById("messages")

    messages.innerHTML += "<p><b>You:</b> " + message + "</p>"

    input.value = ""

    let response = await fetch("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            sessionId: "123",
            message: message
        })
    })

    let data = await response.json()

    console.log(data)

    messages.innerHTML += "<p><b>Assistant:</b> " + data.reply + "</p>"
}