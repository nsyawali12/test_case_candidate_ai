import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Firebase Realtime Chat',
      home: MessagePage(),
    );
  }
}

class MessagePage extends StatefulWidget {
  @override
  _MessagePageState createState() => _MessagePageState();
}

class _MessagePageState extends State<MessagePage> {
  final databaseRef = FirebaseDatabase.instance.ref();
  final TextEditingController messageController = TextEditingController();

  // Function to add new message to Firebase
  Future<void> sendMessage(String message) async {
    try {
      await databaseRef.child('messages').push().set({
        'text': message,
        'timestamp': ServerValue.timestamp,
      });
    } catch (e) {
      // Handle possible errors
      print(e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Real-time Chat"),
      ),
      body: Column(
        children: <Widget>[
          Expanded(
            child: StreamBuilder(
              stream: databaseRef.child('messages').orderByChild('timestamp').onValue,
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  return Text('Error: ${snapshot.error}');
                } else if (snapshot.hasData && !snapshot.hasError) {
                  Event event = snapshot.data! as Event;
                  Map<dynamic, dynamic> messages = event.snapshot.value ?? {};
                  List<Message> messageList = [];
                  messages.forEach((key, value) {
                    final message = Message(
                      text: value['text'],
                      timestamp: value['timestamp'],
                    );
                    messageList.add(message);
                  });
                  messageList.sort((a, b) => b.timestamp.compareTo(a.timestamp));
                  return ListView.builder(
                    itemCount: messageList.length,
                    reverse: true,
                    itemBuilder: (context, index) {
                      return ListTile(
                        title: Text(messageList[index].text),
                        subtitle: Text(DateTime.fromMillisecondsSinceEpoch(
                            messageList[index].timestamp).toString()),
                      );
                    },
                  );
                }
                return Center(child: CircularProgressIndicator());
              },
            ),
          ),
          Padding(
            padding: EdgeInsets.all(8.0),
            child: TextField(
              controller: messageController,
              decoration: InputDecoration(
                labelText: 'Enter message',
                suffixIcon: IconButton(
                  icon: Icon(Icons.send),
                  onPressed: () {
                    sendMessage(messageController.text);
                    messageController.clear();
                  },
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class Message {
  String text;
  int timestamp;

  Message({required this.text, required this.timestamp});
}
