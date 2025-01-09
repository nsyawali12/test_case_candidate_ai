// Custom WordPress shortcode to fetch and display latest posts from a specific category

function custom_recent_posts_shortcode($atts) {
    ob_start();

    $atts = shortcode_atts( array(
        'category' => '-1', // Default category ID if not provided
    ), $atts );

    $category_id = intval($atts['category']);

    global $wpdb;

    $query = $wpdb->prepare( "
        SELECT ID, post_title, post_excerpt, post_date
        FROM $wpdb->posts
        WHERE post_type = 'post' AND post_status = 'publish'
        AND ID IN (
            SELECT object_id
            FROM $wpdb->term_relationships
            WHERE term_taxonomy_id = %d
        )
        ORDER BY post_date DESC
        LIMIT 10",
        $category_id
    );

    $posts = $wpdb->get_results($query);

    if ($posts) {
        echo '<ul>';

        foreach ($posts as $post) {
            echo '<li>';
            echo '<h3>' . $post->post_title . '</h3>';
            echo '<p>' . $post->post_date . '</p>';
            echo '<p>' . $post->post_excerpt . '</p>';
            echo '</li>';
        }

        echo '</ul>';
    } else {
        echo 'No posts found from category ID ' . $category_id;
    }

    return ob_get_clean();
}
add_shortcode('custom_recent_posts', 'custom_recent_posts_shortcode');
