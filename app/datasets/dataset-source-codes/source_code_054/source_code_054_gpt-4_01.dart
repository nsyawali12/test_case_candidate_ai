import 'package:flutter/material.dart';
import 'package:firebase_database/firebase_database.dart';

final databaseReference = FirebaseDatabase.instance.reference();

void main() {
  runApp(MaterialApp(
    home: MessageList(),
  ));
}

class MessageList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Real-time Messages'),
      ),
      body: SafeArea(
          child: Column(
        children: [
          Expanded(
            child: StreamBuilder(
              stream: databaseReference.child("messages").orderByChild('timestamp').onValue,
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  return Center(child: Text('Error: ${snapshot.error}'));
                }
                if (!snapshot.hasData) {
                  return Center(child: Text('Loading...'));
                }

                List<MessageItem> messages = [];

                DataSnapshot data = snapshot.data.snapshot;
                Map<dynamic, dynamic> map = data.value;

                map.forEach((key, value) {
                  messages.insert(
                      0,
                      MessageItem(
                        message: value['message'],
                        timestamp: value['timestamp'],
                      ));
                });

                return ListView.builder(
                  itemCount: messages.length,
                  itemBuilder: (context, index) {
                    return ListTile(
                      title: Text(messages[index].message),
                      subtitle: Text(messages[index].timestamp.toString()),
                    );
                  },
                );
              },
            ),
          ),
          MessageComposer()
        ],
      )),
    );
  }
}

class MessageComposer extends StatefulWidget {
  @override
  MessageComposerState createState() => MessageComposerState();
}

class MessageComposerState extends State<MessageComposer> {
  final TextEditingController _textController = TextEditingController();

  void _handleSubmitted(String text) {
    _textController.clear();
    databaseReference.child("messages").push().set({
      'message': text,
      'timestamp': DateTime.now().millisecondsSinceEpoch
    }).catchError((err) => print(err));
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(horizontal: 8.0),
      child: Row(
        children: <Widget>[
          Flexible(
            child: TextField(
              controller: _textController,
              onSubmitted: _handleSubmitted,
              decoration:
                  InputDecoration.collapsed(hintText: "Send a message"),
            ),
          ),
          FlatButton(
              onPressed: () => _handleSubmitted(_textController.text),
              child: Text("Send")),
        ],
      ),
    );
  }
}

class MessageItem {
  final String message;
  final int timestamp;

  MessageItem({this.message, this.timestamp});
}
