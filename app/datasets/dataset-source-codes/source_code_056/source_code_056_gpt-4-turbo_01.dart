import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MessageCacheManager {
  // The cache duration is set to 30 minutes.
  static const cacheDuration = Duration(minutes: 30);
  List<String>? _cachedMessages;
  DateTime? _lastFetchTime;

  Future<List<String>> fetchMessages() async {
    DateTime now = DateTime.now();

    // Check if we have cached messages and the cache is still valid
    if (_cachedMessages != null && _lastFetchTime != null) {
      if (now.difference(_lastFetchTime!) < cacheDuration) {
        return _cachedMessages!; // Use cached data
      }
    }

    // Fetch new data from Firestore if cache is invalid or not present
    final messages = await _fetchMessagesFromFirestore();
    _cachedMessages = messages;
    _lastFetchTime = now;

    return messages;
  }

  Future<List<String>> _fetchMessagesFromFirestore() async {
    List<String> messages = [];
    final querySnapshot = await FirebaseFirestore.instance.collection('messages').get();

    for (var doc in querySnapshot.docs) {
      messages.add(doc['content']); // Assuming 'content' is the field name for message
    }

    return messages;
  }
}

class MessagesWidget extends StatefulWidget {
  @override
  _MessagesWidgetState createState() => _MessagesWidgetState();
}

class _MessagesWidgetState extends State<MessagesWidget> {
  final _cacheManager = MessageCacheManager();
  List<String>? _messages;

  @override
  void initState() {
    super.initState();
    _loadMessages();
  }

  void _loadMessages() async {
    List<String> messages = await _cacheManager.fetchMessages();
    setState(() {
      _messages = messages;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Messages"),
      ),
      body: _messages == null
          ? CircularProgressIndicator()
          : ListView.builder(
              itemCount: _messages!.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(_messages![index]),
                );
              }),
    );
  }
}

void main() {
  runApp(MaterialApp(home: MessagesWidget()));
}
