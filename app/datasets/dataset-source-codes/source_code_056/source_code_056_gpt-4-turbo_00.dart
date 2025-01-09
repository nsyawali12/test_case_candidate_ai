import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MessageCache {
  DateTime lastFetchTime;
  List<String> cachedMessages;

  MessageCache() {
    lastFetchTime = DateTime.fromMillisecondsSinceEpoch(0); // Initialize to an old date
    cachedMessages = [];
  }

  Future<List<String>> getMessages() async {
    DateTime now = DateTime.now();
    Duration diff = now.difference(lastFetchTime);

    if (cachedMessages.isNotEmpty && diff.inMinutes < 30) {
      // Use the cached data
      return cachedMessages;
    } else {
      // Fetch new data from Firestore
      List<String> messages = await fetchMessagesFromFirestore();
      // Update the cache
      cachedMessages = messages;
      lastFetchTime = DateTime.now();
      return messages;
    }
  }

  Future<List<String>> fetchMessagesFromFirestore() async {
    FirebaseFirestore firestore = FirebaseFirestore.instance;
    QuerySnapshot snapshot = await firestore.collection('messages').get();
    List<String> messages = snapshot.docs.map((doc) => doc['content'] as String).toList();
    return messages;
  }
}

class MessagesWidget extends StatefulWidget {
  @override
  _MessagesWidgetState createState() => _MessagesWidgetState();
}

class _MessagesWidgetState extends State<MessagesWidget> {
  MessageCache messageCache = MessageCache();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Messages'),
      ),
      body: FutureBuilder(
        future: messageCache.getMessages(),
        builder: (BuildContext context, AsyncSnapshot<List<String>> snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            return ListView.builder(
              itemCount: snapshot.data!.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(snapshot.data![index]),
                );
              },
            );
          }
        },
      ),
    );
  }
}
