// Importing necessary libraries
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
      home: MessageApp(),
    );
  }
}

class MessageApp extends StatefulWidget {
  @override
  _MessageAppState createState() => _MessageAppState();
}

class _MessageAppState extends State<MessageApp> {
  // Initializing Firebase Realtime Database
  final dbRef = FirebaseDatabase.instance.reference().child("messages");
  final _textEditingController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    // Using StreamBuilder for real-time updates
    return Scaffold(
      appBar: AppBar(
        title: Text('My Message App'),
      ),
      body: Column(
        children: <Widget>[
          Flexible(
            child: StreamBuilder(
              stream: dbRef.orderByChild('timestamp').onValue,
              builder: (context, snap) {
                if (snap.hasData &&
                    !snap.hasError &&
                    snap.data.snapshot.value != null) {
                  Map data = snap.data.snapshot.value;
                  List item = [];
                  data.forEach((index, data) =>
                      item.add({"key": index, ...data}));

                  // Sorting the messages by timestamp
                  item.sort((a, b) => b["timestamp"].compareTo(a["timestamp"]));
                  return ListView.builder(
                    itemCount: item.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        title: Text(item[index]['message']),
                      );
                    },
                  );
                } else {
                  // Handling potential errors
                  return Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          // Text field to input a new message
          TextField(
            controller: _textEditingController,
            decoration: InputDecoration(
                labelText: 'Enter your message here',
                suffixIcon: IconButton(
                  onPressed: () => submitMessage(_textEditingController.text),
                  icon: Icon(Icons.send),
                )),
          )
        ],
        // Reference the UI content with firebase for every change
        void submitMessage(String message) async {
          // Pushing the new message to the database
          dbRef.push().set({
            "message": message,
            "timestamp": ServerValue.timestamp
          }).catchError((e) {
            print(e);
          });

          // Clear the text field after the message submission
          _textEditingController.clear();
        }
      ),
    );
  }
}
