import 'package:flutter/material.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:firebase_core/firebase_core.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Firebase Messages',
      home: MessagePage(),
    );
  }
}

class MessagePage extends StatefulWidget {
  @override
  _MessagePageState createState() => _MessagePageState();
}

class _MessagePageState extends State<MessagePage> {
  final DatabaseReference _messagesRef = FirebaseDatabase.instance.ref('messages');
  final TextEditingController _messageController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Real-time Messages'),
      ),
      body: Column(
        children: <Widget>[
          Expanded(
            child: StreamBuilder(
              stream: _messagesRef.orderByChild('timestamp').onValue,
              builder: (context, AsyncSnapshot<DatabaseEvent> snapshot) {
                if (snapshot.hasError) {
                  return Text('Error: ${snapshot.error}');
                } else if (snapshot.hasData && snapshot.data!.snapshot.exists) {
                  List<Message> messages = [];
                  Map<dynamic, dynamic> data = snapshot.data!.snapshot.value as Map<dynamic, dynamic>;
                  data.forEach((key, value) {
                    final message = Message.fromMap(value);
                    messages.add(message);
                  });
                  return ListView(
                    children: messages.map((message) => ListTile(
                      title: Text(message.text),
                      subtitle: Text(message.timestamp.toString()),
                    )).toList(),
                  );
                } else {
                  return Center(child: Text('No messages yet'));
                }
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: <Widget>[
                Expanded(
                  child: TextField(
                    controller: _messageController,
                    decoration: InputDecoration(labelText: 'Enter message'),
                  ),
                ),
                SizedBox(width: 8),
                ElevatedButton(
                  onPressed: () => _sendMessage(),
                  child: Text('Send'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _sendMessage() {
    final String messageText = _messageController.text;
    if (messageText.isNotEmpty) {
      final message = Message(
        text: messageText,
        timestamp: DateTime.now().millisecondsSinceEpoch,
      );

      _messagesRef.push().set(message.toJson()).then((_) {
        // Message successfully added to the database
        _messageController.clear();
      }).catchError((error) {
        // Handle any errors here
        print('Error sending message: $error');
      });
    }
  }
}

class Message {
  final String text;
  final int timestamp;

  Message({required this.text, required this.timestamp});

  Map<String, dynamic> toJson() {
    return {
      'text': text,
      'timestamp': timestamp,
    };
  }

  static Message fromMap(Map<dynamic, dynamic> map) {
    return Message(
      text: map['text'],
      timestamp: map['timestamp'],
    );
  }
}
