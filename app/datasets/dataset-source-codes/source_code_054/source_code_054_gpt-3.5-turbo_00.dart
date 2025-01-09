import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_database/firebase_database.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  final databaseRef = FirebaseDatabase.instance.reference().child('messages');

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Real-Time Messages'),
        ),
        body: MessagesList(databaseRef: databaseRef),
        floatingActionButton: AddMessageButton(databaseRef: databaseRef),
      ),
    );
  }
}

class MessagesList extends StatelessWidget {
  final DatabaseReference databaseRef;

  MessagesList({required this.databaseRef});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: databaseRef.orderByChild('timestamp').onValue,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Text('Error: ${snapshot.error}');
        }

        if (!snapshot.hasData) {
          return Center(child: CircularProgressIndicator());
        }

        List<Message> messages = [];
        snapshot.data!.snapshot.value.forEach((key, value) {
          messages.add(Message.fromMap(value));
        });
        messages.sort((a, b) => a.timestamp.compareTo(b.timestamp));

        return ListView.builder(
          itemCount: messages.length,
          itemBuilder: (context, index) {
            return ListTile(
              title: Text(messages[index].text),
            );
          },
        );
      },
    );
  }
}

class AddMessageButton extends StatelessWidget {
  final DatabaseReference databaseRef;

  AddMessageButton({required this.databaseRef});

  @override
  Widget build(BuildContext context) {
    return FloatingActionButton(
      onPressed: () {
        databaseRef.push().set({
          'text': 'New Message',
          'timestamp': DateTime.now().millisecondsSinceEpoch
        });
      },
      child: Icon(Icons.add),
    );
  }
}

class Message {
  final String text;
  final int timestamp;

  Message({required this.text, required this.timestamp});

  factory Message.fromMap(Map<dynamic, dynamic> map) {
    return Message(
      text: map['text'],
      timestamp: map['timestamp'],
    );
  }
}
