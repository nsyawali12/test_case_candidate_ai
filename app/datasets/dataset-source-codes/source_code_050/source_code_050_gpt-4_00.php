function latest_posts_shortcode($atts){
    global $wpdb;

    // Extract attributes
    $a = shortcode_atts(array(
        'catid' => 1
    ), $atts);

    $catid = $a['catid'];

    // Get 10 latest posts from the specific category
    $querystr = $wpdb->prepare("SELECT p.ID, p.post_title, p.post_excerpt, p.post_date 
                                FROM $wpdb->posts p
                                INNER JOIN $wpdb->term_relationships tr ON (p.ID = tr.object_id)
                                INNER JOIN $wpdb->term_taxonomy tt ON (tr.term_taxonomy_id = tt.term_taxonomy_id)
                                WHERE tt.term_id = %d 
                                AND p.post_status = 'publish'
                                ORDER BY p.post_date DESC
                                LIMIT 10", $catid);

    $posts = $wpdb->get_results($querystr, OBJECT);

    $output = '<div class="posts-list">';

    // Loop through each post
    foreach ($posts as $post) {
        $output .= '<div class="post-item">';
        $output .= '<h2 class="post-title">' . $post->post_title . '</h2>';
        $output .= '<p class="post-excerpt">' . $post->post_excerpt . '</p>';
        $output .= '<p class="post-date">' . date('F j, Y', strtotime($post->post_date)) . '</p>';
        $output .= '</div>';
    }

    $output .= '</div>';

    return $output;
}
add_shortcode('latest_posts', 'latest_posts_shortcode');
