// JavaScript Document
<!-- Users script: A very simple usage of JQuery library -->
<!-- My simple test on JQuery library-->


  // entry point for the javascript function...
  //  javascript interaction logic - powered by JQuery
  
        
 $(document).ready(function(){

   // default operations
   // id naming convention
   // menuXXX - id of menu items
   // blockXXX - id of blocks
   {
   
      
   
   $(".showall-solution").bind('click', function(e) { $(".solution-content").show(); });
   $(".hideall-solution").bind('click', function(e) { $(".solution-content").hide(); });
                                
                        // question-content class's subling with class solution
    $(".question-content ~ .toggle-solution").bind('click', function(e) {
                                                                
                                                        
                // may have multiple obj block here...
                var objList = $(this).siblings(".solution-content");
                                                                                                
                //alert(objList.size());
                // MUST use div!!
                                        
                objList.each( function(i) {
                        if ( $(this).css("display") == "block" )
                        {
                                $(this).hide("slow");
                        }
                        else
                        {
                                $(this).show("slow");
                        }
                                                                                                        
                });
                                                                                
                        
        } );
                                
        enableTooltips(); // script for bubble tooltips
        
        // scripts for jQuery calender
        $("#calendarbar").calendar({onSelect: updateInlineCalendar,
          customDate: specialCOMP2011Days });
        
        
        $("#block0").siblings().hide();$("#block0").show();<!--INITIALIZE-SCRIPT-->
        
   }

   // Toggle between two views...
   
   $("#tabShowAllOnePage").click( function() {
                var w = window.open();
                var html = $("#main").html();
		w.document.writeln(html);
                $("#block0").siblings().show();
                $("#block0").show();
                $("#sidebar").hide();
                $("#main").css("width","100%");
                $("#footer").hide();
        
                $(".toggle-solution").hide();
                $(".solution-content").show();  // also show all solutions...
                
                $("iframe").hide(); // hide all iframe slides...
   }
   );
   
   $("#tabShowNormalView").click( function() {
                $("#block0").siblings().hide();
                $("#block0").show();
                $("#sidebar").show();           
                $("#main").css("width","730px");
                $("#footer").show();            
                
                
                $(".toggle-solution").show(); // show all toggle buttons
                
                $("iframe").show(); // show all iframe slides...
                
   }
   );
   
   // font size manipulation
   $("#tabIncFontSize").click(function(){
                var s = $("#main").css("font-size");
                s = parseInt(s.substring(0, s.length-2));
                s = s+1;
                if (s>18) return false;
                $("#main").css("font-size", s+"px");
        }
   );
   
   $("#tabDecFontSize").click(function(){
                var s = $("#main").css("font-size");
                s = parseInt(s.substring(0, s.length-2));
                s = s-1;
                if (s<10) return false;
                $("#main").css("font-size",s+"px");
        }
   );

   
   
  /* 
$("#menu0").click(function(){
$("#block0").siblings().hide();
$("#block0").show();
});
*/
$("#menu1").click(function(){
$("#block1").siblings().hide();
$("#block1").show();
});

$("#menu2").click(function(){
$("#block2").siblings().hide();
$("#block2").show();
});

/*$("#menu3").click(function(){
$("#block3").siblings().hide();
$("#block3").show();
});

$("#menu4").click(function(){
$("#block4").siblings().hide();
$("#block4").show();
});

$("#menu5").click(function(){
$("#block5").siblings().hide();
$("#block5").show();
});

$("#menu6").click(function(){
$("#block6").siblings().hide();
$("#block6").show();
});

$("#menu7").click(function(){
$("#block7").siblings().hide();
$("#block7").show();
});

$("#menu8").click(function(){
$("#block8").siblings().hide();
$("#block8").show();
});

$("#menu9").click(function(){
$("#block9").siblings().hide();
$("#block9").show();
});

$("#menu10").click(function(){
$("#block10").siblings().hide();
$("#block10").show();
});

$("#menu11").click(function(){
$("#block11").siblings().hide();
$("#block11").show();
});

$("#menu12").click(function(){
$("#block12").siblings().hide();
$("#block12").show();
});
*/

<!--JAVASCRIPT-MENU-OPERATION-->
});
 
         <!-- JQuery calender logics -->
         function updateInlineCalendar() {
                                
                // this callback is called when the select option is changed.
                //inlineDate = $('#calendarbar');
                //var dateSelected = popUpCal.getDateFor(inlineDate[0]);
                //alert(dateSelected);
         }
  
        //var natDays = [[10, 6], [10, 15], [11, 3], [11, 17], [12, 8], [12, 18], [12,19]];
        var natDays = [ ] ;

        function specialCOMP2011Days(date) {
                for (i = 0; i < natDays.length; i++) {
                        if (date.getMonth() == natDays[i][0] - 1 && date.getDate() == natDays[i][1]) {
                                return [true, 'special_day'];
                        }
                }
                return [true, ''];
        }

function write_countdown(deadline)
{
        var today=new Date();
        // comment out, makes it as a function of date...
        // var deadline = new Date(2008,10-1,04,23,59); // month - count from 0-11      
        var timeDiff = deadline.getTime() - today.getTime();
        if (timeDiff < 0)
        {
                document.write("The deadline is passed!");
        }
        else
        {
                var one_minute = 1000*60 ;
                var one_hour = one_minute*60 ;
                var one_day = one_hour*24;
                                                
                var remainTime = timeDiff;              
                var nDays = Math.floor(timeDiff/one_day);
                remainTime = remainTime - nDays*one_day;
                var nHours = Math.floor( remainTime/one_hour );
                remainTime = remainTime - nHours*one_hour ;
                var nMinutes = Math.floor( remainTime/one_minute );
                
                document.write("<font color='red'>");
                document.write(nDays+" days "+nHours+" hours "+nMinutes+" minutes" + " before the deadline");           
                document.write("</font>");
        }
}
