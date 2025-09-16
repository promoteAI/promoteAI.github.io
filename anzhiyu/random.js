var posts=["2023/05/12/sheng-cheng-shi-ai-xiang-jie/"];function toRandomPost(){
    pjax.loadUrl('/'+posts[Math.floor(Math.random() * posts.length)]);
  };