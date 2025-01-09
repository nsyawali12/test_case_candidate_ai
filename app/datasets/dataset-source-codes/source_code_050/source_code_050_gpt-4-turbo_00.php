function fetch_latest_posts_by_category($atts) {
    // Extract category ID from shortcode attributes
    $atts = shortcode_atts(array(
        'cat_id' => '1', // default category ID
    ), $atts);

    // Define and execute the SQL query to get the latest 10 posts from the specified category
    global $wpdb;
    $query = $wpdb->prepare(
        "SELECT p.ID, p.post_title, p.post_excerpt, p.post_date
        FROM $wpdb->posts p
        INNER JOIN $wpdb->term_relationships tr ON p.ID = tr.object_id
        INNER JOIN $wpdb->term_taxonomy tt ON tr.term_taxonomy_id = tt.term_taxonomy_id
        WHERE tt.taxonomy = 'category' AND tt.term_id = %d
        AND p.post_status = 'publish'
        ORDER BY p.post_date DESC LIMIT 10",
        $atts['cat_id']
    );
    $posts = $wpdb->get_results($query);

    // Start building the output
    $output = '<ul class="latest-posts">';
    foreach ($posts as $post) {
        $post_date = date("F j, Y", strtotime($post->post_date)); // Format the date
        $output .= sprintf(
            '<li><h2>%s</h2><p>%s</p><small>%s</small></li>',
            esc_html($post->post_title),
            esc_html($post->post_excerpt),
            esc_html($post_date)
        );
    }
    $output .= '</ul>';

    // Add some basic CSS for the output
    $output .= '
    <style>
        .latest-posts li {
            list-style: none;
            padding: 10px;
            border-bottom: 1px solid #ccc;
        }
        .latest-posts h2 {
            margin: 0 0 5px 0;
        }
        .latest-posts p {
            margin: 0;
        }
        .latest-posts small {
            color: #666;
        }
    </style>';

    return $output;
}

// Register the shortcode [latest_posts cat_id="X"]
add_shortcode('latest_posts', 'fetch_latest_posts_by_category');
