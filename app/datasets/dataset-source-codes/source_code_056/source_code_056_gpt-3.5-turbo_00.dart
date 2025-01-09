import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MessageCache extends ChangeNotifier {
  List<Message> _cachedMessages = [];
  DateTime _lastFetchTime;

  Future<void> fetchMessages() async {
    if (_lastFetchTime == null || DateTime.now().difference(_lastFetchTime) >= Duration(minutes: 30)) {
      QuerySnapshot snapshot = await FirebaseFirestore.instance.collection('messages').get();
      _cachedMessages = snapshot.docs.map((doc) => Message.fromFirestore(doc)).toList();
      _lastFetchTime = DateTime.now();
      notifyListeners();
    }
  }

  List<Message> get cachedMessages => List.unmodifiable(_cachedMessages);
}

class Message {
  final String id;
  final String content;

  Message({this.id, this.content});

  factory Message.fromFirestore(DocumentSnapshot doc) {
    return Message(
      id: doc.id,
      content: doc.data()['content'],
    );
  }
}

// Example usage:
void main() {
  MessageCache messageCache = MessageCache();

  // Simulate new fetch attempt at 10:25 AM
  _updateCacheIfNeeded(messageCache);

  // Simulate new fetch attempt at 10:35 AM
  _updateCacheIfNeeded(messageCache);
}

void _updateCacheIfNeeded(MessageCache cache) async {
  await cache.fetchMessages();
  print('Cached messages: ${cache.cachedMessages}');
}

