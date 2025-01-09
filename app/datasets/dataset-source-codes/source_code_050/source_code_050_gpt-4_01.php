function latest_posts_func($atts) {
    // Define the shortcode attribute
    $atts = shortcode_atts(array(
        'cat_id' => '1', // Default value for category ID
    ), $atts, 'latest_posts');

    global $wpdb;
    $cat_id = (int) $atts['cat_id'];
    
    // Fetch the latest 10 posts for a specific category using SQL
    $results = $wpdb->get_results("SELECT wp_posts.* FROM wp_posts
    LEFT JOIN wp_term_relationships on (wp_posts.ID = wp_term_relationships.object_id)
    LEFT JOIN wp_term_taxonomy on (wp_term_relationships.term_taxonomy_id = wp_term_taxonomy.term_taxonomy_id)
    WHERE wp_term_taxonomy.term_id=$cat_id AND wp_posts.post_status = 'publish'
    ORDER BY wp_posts.post_date DESC LIMIT 10");

    // Check if we got results from the database
    if (count($results) > 0) {
        // Start the unordered list
        $output = '<ul style="list-style-type: none;">';
        
        // Loop through each post
        foreach ($results as $post) {
            $output.= '<li style="padding: 15px 0; border-bottom: 1px dashed #ddd">';
            // Add title, date and excerpt into the list
            $output.= '<h2>'. $post->post_title .'</h2>';
            $output.= '<p>'. $post->post_date .'</p>';
            $output.= '<p>'. wp_trim_excerpt($post->post_excerpt) .'</p>';
            $output.= '</li>';
        }
        
        // Close the unordered list
        $output.= '</ul>';
        
        return $output;
    }
}
// Register the shortcode [latest_posts]
add_shortcode( 'latest_posts', 'latest_posts_func' );
