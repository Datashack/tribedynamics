
//3D EFFECT
//A PEN BY Dennis Garrn: https://codepen.io/dennisgarrn/pen/kHEKn
jQuery(document).ready(function () {
    $('h1').mousemove(function (e) {
        var rXP = (e.pageX - this.offsetLeft - $(this).width() / 2);
        var rYP = (e.pageY - this.offsetTop - $(this).height() / 2);
        $('h1').css('text-shadow', +rYP / 10 + 'px ' + rXP / 80 + 'px rgba(32,65,123,1), ' + rYP / 8 + 'px ' + rXP / 60 + 'px rgba(60,188,188,1), ' + rXP / 70 + 'px ' + rYP / 12 + 'px rgba(64,255,211,1)');
    });
});



