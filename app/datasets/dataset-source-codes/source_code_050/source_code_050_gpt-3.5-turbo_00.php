// Register custom shortcode to fetch and display latest posts from a specific category
function custom_category_posts_shortcode($atts) {
    // Extract shortcode attributes
    extract(shortcode_atts(array(
        'category_id' => 0, // Default category ID is 0
    ), $atts));

    global $wpdb;

    // Query to fetch latest 10 posts from specific category using SQL
    $query = $wpdb->prepare("
        SELECT p.post_title, p.post_excerpt, p.post_date
        FROM {$wpdb->prefix}posts AS p
        INNER JOIN {$wpdb->prefix}term_relationships AS tr ON p.ID = tr.object_id
        INNER JOIN {$wpdb->prefix}term_taxonomy AS tt ON tr.term_taxonomy_id = tt.term_taxonomy_id
        WHERE p.post_type = 'post' AND p.post_status = 'publish' AND tt.term_id = %d
        ORDER BY p.post_date DESC
        LIMIT 10
    ", $category_id);

    $posts = $wpdb->get_results($query);

    // Build HTML output for the posts
    $output = '<ul class="custom-category-posts">';

    foreach ($posts as $post) {
        $output .= '<li>';
        $output .= '<h3>' . $post->post_title . '</h3>';
        $output .= '<p>' . $post->post_excerpt . '</p>';
        $output .= '<p>' . $post->post_date . '</p>';
        $output .= '</li>';
    }

    $output .= '</ul>';

    return $output;
}

// Register the shortcode for easy use in posts and pages
add_shortcode('custom_posts_category', 'custom_category_posts_shortcode');
