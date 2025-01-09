<?php
// Add the following code to your theme's functions.php

// Shortcode to fetch latest posts by category
function fetch_latest_posts_by_category($atts) {
    // Shortcode attributes and default category
    $atts = shortcode_atts(array('category_id' => '1'), $atts);

    // Get category ID from shortcode attribute
    $category_id = $atts['category_id'];

    // SQL to fetch the latest 10 posts from a specific category
    global $wpdb;
    $query = "
        SELECT p.ID, p.post_title, p.post_date, p.post_excerpt
        FROM {$wpdb->posts} AS p
        INNER JOIN {$wpdb->term_relationships} AS tr ON p.ID = tr.object_id
        WHERE tr.term_taxonomy_id = %d
        AND p.post_status = 'publish'
        AND p.post_type = 'post'
        ORDER BY p.post_date DESC
        LIMIT 10
    ";

    // Prepare and execute the query
    $posts = $wpdb->get_results($wpdb->prepare($query, $category_id));

    // Start output buffering
    ob_start();

    // HTML Output to display the posts
    echo '<ul class="latest-posts">';
    foreach ($posts as $post) {
        $post_date = date('F j, Y', strtotime($post->post_date));
        echo '<li>';
        echo '<h3>' . esc_html($post->post_title) . '</h3>';
        echo '<p class="excerpt">' . esc_html($post->post_excerpt) . '</p>';
        echo '<p class="date">' . esc_html($post_date) . '</p>';
        echo '</li>';
    }
    echo '</ul>';

    // Retrieve output from buffer and return
    return ob_get_clean();
}

// Register the shortcode
add_shortcode('latest_posts_by_category', 'fetch_latest_posts_by_category');

// Register styles
function latest_posts_styles() {
    ?>
    <style>
        .latest-posts li {
            margin-bottom: 20px;
            list-style-type: none;
        }

        .latest-posts h3 {
            margin: 0 0 10px;
        }

        .latest-posts .excerpt {
            font-size: 14px;
            color: #666;
            margin: 0;
        }

        .latest-posts .date {
            font-size: 12px;
            color: #999;
        }
    </style>
    <?php
}

// Enqueue styles in the head
add_action('wp_head', 'latest_posts_styles');
?>
