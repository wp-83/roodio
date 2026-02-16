<?php

namespace Database\Seeders;

use App\Models\Songs;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class SongsSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Mood ID mapping
        $moodIds = [
            'angry' => 'MD-0000004',
            'happy' => 'MD-0000001',
            'relaxed' => 'MD-0000003',
            'sad' => 'MD-0000002',
        ];

        // User ID for seeding (adjust if needed)
        $userId = 'US-0000001';

        // Songs data - 179 songs total
        $songs = [
            // ANGRY SONGS (47 songs - removed 3 missing)
            ['mood' => 'angry', 'title' => 'Living Hell', 'artist' => 'Bella Poarch', 'lyrics' => 'Living Hell lyrics...', 'filename' => '1_Bella Poarch - Living Hell (Official Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'Happier Than Ever', 'artist' => 'Billie Eilish', 'lyrics' => 'Happier Than Ever lyrics...', 'filename' => '2_Billie Eilish - Happier Than Ever.mp3'],
            ['mood' => 'angry', 'title' => 'Cry', 'artist' => 'Connor Kauffman', 'lyrics' => 'Cry lyrics...', 'filename' => '3_Connor Kauffman - Cry (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Crying while you\'re dancing', 'artist' => 'Dayseeker', 'lyrics' => 'Crying while you\'re dancing lyrics...', 'filename' => '4_Dayseeker - Crying while you\'re dancing (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'quiet', 'artist' => 'eli.', 'lyrics' => 'quiet lyrics...', 'filename' => '5_eli. - quiet. (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Falling Through Petals', 'artist' => 'Various Artists', 'lyrics' => 'Falling Through Petals lyrics...', 'filename' => '6_Falling Through Petals.mp3'],
            ['mood' => 'angry', 'title' => 'Tears of Gold', 'artist' => 'Faouzia', 'lyrics' => 'Tears of Gold lyrics...', 'filename' => '7_Faouzia - Tears of Gold (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Fire Alarm', 'artist' => 'Soffie Dossi', 'lyrics' => 'Fire Alarm lyrics...', 'filename' => '8_Fire Alarm - Soffie Dossi (lyrics).mp3'],
            // REMOVED: God, Save Me From Myself
            ['mood' => 'angry', 'title' => 'How Was I Supposed to Know', 'artist' => 'Various Artists', 'lyrics' => 'How Was I Supposed to Know lyrics...', 'filename' => '10_How Was I Supposed to Know (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Ich geb\'s aus', 'artist' => 'Various Artists', 'lyrics' => 'Ich geb\'s aus lyrics...', 'filename' => '11_Ich geb\'s aus.mp3'],
            ['mood' => 'angry', 'title' => 'Murder My Feelings', 'artist' => 'Lala Sadii', 'lyrics' => 'Murder My Feelings lyrics...', 'filename' => '12_Murder My Feelings (Lyrics) - Lala Sadii.mp3'],
            ['mood' => 'angry', 'title' => 'Tattoos', 'artist' => 'Natalie Jane', 'lyrics' => 'Tattoos lyrics...', 'filename' => '13_Natalie Jane - Tattoos (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'I\'m not okay', 'artist' => 'Various Artists', 'lyrics' => 'I\'m not okay lyrics...', 'filename' => '14_I\'m not okay.mp3'],
            ['mood' => 'angry', 'title' => 'Some Things', 'artist' => 'Nevertel', 'lyrics' => 'Some Things lyrics...', 'filename' => '15_Nevertel - Some Things [Lyrics].mp3'],
            ['mood' => 'angry', 'title' => 'lost in the moment', 'artist' => 'NF', 'lyrics' => 'lost in the moment lyrics...', 'filename' => '16_NF- lost in the moment- lyrics.mp3'],
            ['mood' => 'angry', 'title' => 'Only When Its You', 'artist' => 'Various Artists', 'lyrics' => 'Only When Its You lyrics...', 'filename' => '17_Official Lyric video for Only When Its You.mp3'],
            ['mood' => 'angry', 'title' => 'jealousy, jealousy', 'artist' => 'Olivia Rodrigo', 'lyrics' => 'jealousy, jealousy lyrics...', 'filename' => '18_Olivia Rodrigo - jealousy, jealousy (Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'vampire', 'artist' => 'Olivia Rodrigo', 'lyrics' => 'vampire lyrics...', 'filename' => '19_Olivia Rodrigo - vampire.mp3'],
            ['mood' => 'angry', 'title' => 'You\'re Not Sorry', 'artist' => 'Taylor Swift', 'lyrics' => 'You\'re Not Sorry lyrics...', 'filename' => '20_Taylor Swift - You\'re Not Sorry (Taylor\'s Version) (Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'because i liked a boy', 'artist' => 'Sabrina Carpenter', 'lyrics' => 'because i liked a boy lyrics...', 'filename' => '21_Sabrina Carpenter - because i liked a boy (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Hereditary', 'artist' => 'Wind Walker', 'lyrics' => 'Hereditary lyrics...', 'filename' => '22_Wind Walker - Hereditary (Official Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'I Was Only Afraid', 'artist' => 'Revenge In Kyoto', 'lyrics' => 'I Was Only Afraid lyrics...', 'filename' => '23_Revenge In Kyoto - I Was Only Afraid Lyric Video.mp3'],
            ['mood' => 'angry', 'title' => 'You', 'artist' => 'Various Artists', 'lyrics' => 'You lyrics...', 'filename' => '24_You.mp3'],
            ['mood' => 'angry', 'title' => 'The Journey', 'artist' => 'Sik World', 'lyrics' => 'The Journey lyrics...', 'filename' => '25_Sik World - The Journey.mp3'],

            ['mood' => 'angry', 'title' => 'Just Pretend', 'artist' => 'BAD OMENS', 'lyrics' => 'Just Pretend lyrics...', 'filename' => '26_BAD OMENS - Just Pretend (Official Music Video).mp3'],
            ['mood' => 'angry', 'title' => 'Therefore I Am', 'artist' => 'Billie Eilish', 'lyrics' => 'Therefore I Am lyrics...', 'filename' => '27_Billie Eilish - Therefore I Am (Official Music Video).mp3'],
            ['mood' => 'angry', 'title' => 'Burning Down', 'artist' => 'Alex', 'lyrics' => 'Burning Down lyrics...', 'filename' => '28_Burning Down (Alexs Version).mp3'],
            ['mood' => 'angry', 'title' => 'Thank You for Hating Me', 'artist' => 'Citizen Soldier', 'lyrics' => 'Thank You for Hating Me lyrics...', 'filename' => '29_Citizen Soldier - Thank You for Hating Me (Official Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'The Sound Of Silence', 'artist' => 'Disturbed', 'lyrics' => 'The Sound Of Silence lyrics...', 'filename' => '30_Disturbed - The Sound Of Silence (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Saints', 'artist' => 'Echos', 'lyrics' => 'Saints lyrics...', 'filename' => '31_Echos - Saints (Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'cinderella\'s dead', 'artist' => 'EMELINE', 'lyrics' => 'cinderella\'s dead lyrics...', 'filename' => '32_EMELINE - cinderella\'s dead (Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'Gasoline', 'artist' => 'Halsey', 'lyrics' => 'Gasoline lyrics...', 'filename' => '33_Halsey - Gasoline (Audio).mp3'],
            ['mood' => 'angry', 'title' => 'happy ending', 'artist' => 'bailey spinn', 'lyrics' => 'happy ending lyrics...', 'filename' => '34_happy ending - bailey spinn [Official Music Video].mp3'],
            ['mood' => 'angry', 'title' => 'I asked for so little', 'artist' => 'Various Artists', 'lyrics' => 'I asked for so little lyrics...', 'filename' => '35_I asked for so little LYRICS.mp3'],
            ['mood' => 'angry', 'title' => 'He\'s A 10', 'artist' => 'JESSIA', 'lyrics' => 'He\'s A 10 lyrics...', 'filename' => '36_JESSIA - He\'s A 10 (Official Lyric Video).mp3'],
            ['mood' => 'angry', 'title' => 'excuses', 'artist' => 'Jessica Baio', 'lyrics' => 'excuses lyrics...', 'filename' => '37_Jessica Baio - excuses (official audio).mp3'],
            ['mood' => 'angry', 'title' => 'Im not there for you', 'artist' => 'Jessie Murph', 'lyrics' => 'Im not there for you lyrics...', 'filename' => '38_Jessie Murph - Im not there for you (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'While You\'re At It', 'artist' => 'Jessie Murph', 'lyrics' => 'While You\'re At It lyrics...', 'filename' => '39_Jessie Murph - While You\'re At It (Official Video).mp3'],
            // REMOVED: Lauren Spencer Smith - Lighting the flame
            ['mood' => 'angry', 'title' => '10 Things I Hate About You', 'artist' => 'Leah Kate', 'lyrics' => '10 Things I Hate About You lyrics...', 'filename' => '41_Leah Kate - 10 Things I Hate About You (Official Music Video).mp3'],
            ['mood' => 'angry', 'title' => 'Architect', 'artist' => 'Livingston', 'lyrics' => 'Architect lyrics...', 'filename' => '42_Livingston - Architect (Lyrics).mp3'],
            ['mood' => 'angry', 'title' => 'Class of 2013', 'artist' => 'Mitski', 'lyrics' => 'Class of 2013 lyrics...', 'filename' => '43_Mitski - Class of 2013 (Audiotree Live).mp3'],
            ['mood' => 'angry', 'title' => 'fallin', 'artist' => 'Natalie Jane', 'lyrics' => 'fallin lyrics...', 'filename' => '44_Natalie Jane - fallin.mp3'],
            ['mood' => 'angry', 'title' => 'Therapy Session', 'artist' => 'NF', 'lyrics' => 'Therapy Session lyrics...', 'filename' => '45_NF - Therapy Session.mp3'],
            ['mood' => 'angry', 'title' => 'ANXIOUS', 'artist' => 'sami rose', 'lyrics' => 'ANXIOUS lyrics...', 'filename' => '46_sami rose - ANXIOUS (Official Video).mp3'],
            ['mood' => 'angry', 'title' => 'Everybody Supports Women', 'artist' => 'SOFIA ISELLA', 'lyrics' => 'Everybody Supports Women lyrics...', 'filename' => '47_SOFIA ISELLA - Everybody Supports Women (Official Music Video).mp3'],
            ['mood' => 'angry', 'title' => 'I Hate U', 'artist' => 'SZA', 'lyrics' => 'I Hate U lyrics...', 'filename' => '48_SZA - I Hate U (Audio).mp3'],
            ['mood' => 'angry', 'title' => 'don\'t come back', 'artist' => 'Tate McRae', 'lyrics' => 'don\'t come back lyrics...', 'filename' => '49_Tate McRae - don\'t come back (Lyric Video).mp3'],
            // REMOVED: Taylor Swift mad woman

            // RELAXED SONGS (35 songs - removed 15 missing)
            // REMOVED: Ardhito Pramono - I Just Couldnt Save You Tonight
            ['mood' => 'relaxed', 'title' => 'Call of Silence', 'artist' => 'Attack On Titan OST', 'lyrics' => 'Call of Silence lyrics...', 'filename' => '52_Attack On Titan OST- Call of Silence (Lyrics) - ThatOnePotato.mp3'],
            ['mood' => 'relaxed', 'title' => 'Duvet', 'artist' => 'Bôa', 'lyrics' => 'Duvet lyrics...', 'filename' => '53_Bôa - Duvet (Lyrics) - Taj Tracks.mp3'],
            ['mood' => 'relaxed', 'title' => 'Bags', 'artist' => 'Clairo', 'lyrics' => 'Bags lyrics...', 'filename' => '54_Clairo - Bags (Lyrics) - The Tiny Majority.mp3'],
            ['mood' => 'relaxed', 'title' => 'Always', 'artist' => 'Daniel Caesar', 'lyrics' => 'Always lyrics...', 'filename' => '55_Daniel Caesar - Always (Lyrics) - 7clouds Rock.mp3'],
            ['mood' => 'relaxed', 'title' => 'Get You', 'artist' => 'Daniel Caesar ft. Kali Uchis', 'lyrics' => 'Get You lyrics...', 'filename' => '56_Daniel Caesar - Get You (Lyrics) ft. Kali Uchis - Luxury Tones.mp3'],
            ['mood' => 'relaxed', 'title' => 'Kingston', 'artist' => 'Faye Webster', 'lyrics' => 'Kingston lyrics...', 'filename' => '57_Faye Webster Kingston (Lyrics) - Worldly Hits.mp3'],
            ['mood' => 'relaxed', 'title' => 'Godspeed', 'artist' => 'Frank Ocean', 'lyrics' => 'Godspeed lyrics...', 'filename' => '58_Frank Ocean - Godspeed (Lyrics) - Lyricz.mp3'],
            ['mood' => 'relaxed', 'title' => 'I Thought I Saw Your Face Today', 'artist' => 'She And Him', 'lyrics' => 'I Thought I Saw Your Face Today lyrics...', 'filename' => '59_I Thought I Saw Your Face Today - She And Him (Lyrics) - Cassiopeia.mp3'],
            ['mood' => 'relaxed', 'title' => 'SLOW DANCING IN THE DARK', 'artist' => 'Joji', 'lyrics' => 'SLOW DANCING IN THE DARK lyrics...', 'filename' => '60_Joji - SLOW DANCING IN THE DARK (Lyrics) - 7clouds.mp3'],
            ['mood' => 'relaxed', 'title' => 'Romantic Homicide', 'artist' => 'd4vd', 'lyrics' => 'Romantic Homicide lyrics...', 'filename' => '61_d4vd - Romantic Homicide (Lyrics) - 7clouds.mp3'],
            ['mood' => 'relaxed', 'title' => 'innocence', 'artist' => 'HildirSvensson & Six dior', 'lyrics' => 'innocence lyrics...', 'filename' => '62_innocence- HildirSvensson & Six dior. (Lyrics) - yunooyle.mp3'],
            ['mood' => 'relaxed', 'title' => 'her', 'artist' => 'JVKE', 'lyrics' => 'her lyrics...', 'filename' => '63_JVKE - her (official lyric video) - JVKE.mp3'],
            ['mood' => 'relaxed', 'title' => 'WANTCHU', 'artist' => 'keshi', 'lyrics' => 'WANTCHU lyrics...', 'filename' => '64_keshi - WANTCHU (Lyrics) - Urban Paradise.mp3'],
            ['mood' => 'relaxed', 'title' => 'Let Down', 'artist' => 'Radiohead', 'lyrics' => 'Let Down lyrics...', 'filename' => '65_Let Down - Radiohead - Lyrics - Pyroarp.mp3'],
            ['mood' => 'relaxed', 'title' => 'Chest Pain', 'artist' => 'Malcolm Todd', 'lyrics' => 'Chest Pain lyrics...', 'filename' => '66_Malcolm Todd - Chest Pain (I Love) (Lyrics) - 7clouds.mp3'],
            ['mood' => 'relaxed', 'title' => 'Multo', 'artist' => 'Cup of Joe', 'lyrics' => 'Multo lyrics...', 'filename' => '67_Multo — Cup of Joe (Lyrics) - cezca.mp3'],
            ['mood' => 'relaxed', 'title' => 'lowkey', 'artist' => 'NIKI', 'lyrics' => 'lowkey lyrics...', 'filename' => '68_NIKI - lowkey (Lyrics) - Dan Music.mp3'],
            ['mood' => 'relaxed', 'title' => 'BEAUTiFUL', 'artist' => 'Paul Partohap', 'lyrics' => 'BEAUTiFUL lyrics...', 'filename' => '69_Paul Partohap - BEAUTiFUL (Lyric Video) - paul partohap.mp3'],
            ['mood' => 'relaxed', 'title' => 'Line Without a Hook', 'artist' => 'Ricky Montgomery', 'lyrics' => 'Line Without a Hook lyrics...', 'filename' => '70_Ricky Montgomery - Line Without a Hook (Official Lyric Video) - Ricky Montgomery.mp3'],
            ['mood' => 'relaxed', 'title' => 'THANK YOU 4 LOVIN\' ME', 'artist' => 'Paul Partohap', 'lyrics' => 'THANK YOU 4 LOVIN\' ME lyrics...', 'filename' => '71_THANK YOU 4 LOVIN\' ME -  Paul Partohap (lirik video) - AnyTime Music.mp3'],
            ['mood' => 'relaxed', 'title' => 'I Love You So', 'artist' => 'The Walters', 'lyrics' => 'I Love You So lyrics...', 'filename' => '72_The Walters - I Love You So (Lyrics) - Vibe Music.mp3'],
            ['mood' => 'relaxed', 'title' => 'Answer', 'artist' => 'Tyler, The Creator', 'lyrics' => 'Answer lyrics...', 'filename' => '73_Tyler, The Creator - Answer (Lyrics) - Rap City.mp3'],
            ['mood' => 'relaxed', 'title' => 'bad', 'artist' => 'wave to earth', 'lyrics' => 'bad lyrics...', 'filename' => '74_wave to earth - bad (Lyrics) - Alternate.mp3'],
            ['mood' => 'relaxed', 'title' => 'love.', 'artist' => 'wave to earth', 'lyrics' => 'love. lyrics...', 'filename' => '75_wave to earth - love. (Lyrics) [HAN_ROM_ENG] - hazelnut latte.mp3'],
            // REMOVED: beabadoobee-Coffee_
            // REMOVED: Coldplay-The Scientist
            // REMOVED: Everybody Loves An Outlaw - I See Red
            // REMOVED: Forest-Blakk If You Love Her
            // REMOVED: Goo-Goo Dolls Iris
            // REMOVED: Henry Moodie - pick up the phone
            // REMOVED: Jamie Miller - Heres Your Perfect
            ['mood' => 'relaxed', 'title' => 'Cinnamon Girl', 'artist' => 'Lana Del Rey', 'lyrics' => 'Cinnamon Girl lyrics...', 'filename' => '83_Lana Del Rey - Cinnamon Girl.mp3'],
            ['mood' => 'relaxed', 'title' => 'Radio', 'artist' => 'Lana Del Rey', 'lyrics' => 'Radio lyrics...', 'filename' => '84_Lana Del Rey - Radio.mp3'],
            // REMOVED: laufey-From The Start
            ['mood' => 'relaxed', 'title' => 'bittersweet', 'artist' => 'Madison Beer', 'lyrics' => 'bittersweet lyrics...', 'filename' => '86_Madison Beer - bittersweet.mp3'],
            ['mood' => 'relaxed', 'title' => 'Reckless', 'artist' => 'Madison Beer', 'lyrics' => 'Reckless lyrics...', 'filename' => '87_Madison Beer - Reckless.mp3'],
            ['mood' => 'relaxed', 'title' => 'Night Changes', 'artist' => 'One Direction', 'lyrics' => 'Night Changes lyrics...', 'filename' => '88_Night Changes - One Direction.mp3'],
            ['mood' => 'relaxed', 'title' => 'You\'ll Be in My Heart', 'artist' => 'NIKI ft. Phil Collins', 'lyrics' => 'You\'ll Be in My Heart lyrics...', 'filename' => '89_NIKI - You\'ll Be in My Heart ft. Phil Collins.mp3'],
            ['mood' => 'relaxed', 'title' => '18', 'artist' => 'One Direction', 'lyrics' => '18 lyrics...', 'filename' => '90_One Direction - 18.mp3'],
            // REMOVED: Ravyn Lenae - Love Me Not
            // REMOVED: Shawn Mendes - Mercy
            // REMOVED: sombr - back to friends
            ['mood' => 'relaxed', 'title' => 'Until I Found You', 'artist' => 'Stephen Sanchez', 'lyrics' => 'Until I Found You lyrics...', 'filename' => '94_Stephen Sanchez - Until I Found You.mp3'],
            ['mood' => 'relaxed', 'title' => 'Nobody Gets Me', 'artist' => 'SZA', 'lyrics' => 'Nobody Gets Me lyrics...', 'filename' => '95_SZA - Nobody Gets Me.mp3'],
            ['mood' => 'relaxed', 'title' => 'Eldest Daughter', 'artist' => 'Taylor Swift', 'lyrics' => 'Eldest Daughter lyrics...', 'filename' => '96_Taylor Swift - Eldest Daughter.mp3'],
            ['mood' => 'relaxed', 'title' => 'Snow On The Beach', 'artist' => 'Taylor Swift ft. Lana Del Rey', 'lyrics' => 'Snow On The Beach lyrics...', 'filename' => '97_Taylor Swift - Snow On The Beach ft. Lana Del Rey.mp3'],
            // REMOVED: Teddy Swims - Lose Control
            // REMOVED: The 1975 - About You
            ['mood' => 'relaxed', 'title' => 'The Man Who Can\'t Be Moved', 'artist' => 'The Script', 'lyrics' => 'The Man Who Can\'t Be Moved lyrics...', 'filename' => '100_The Script - The Man Who Can\'t Be Moved.mp3'],

            // HAPPY SONGS (48 songs - removed 2 missing)
            ['mood' => 'happy', 'title' => '2002', 'artist' => 'Anne Marie', 'lyrics' => '2002 lyrics...', 'filename' => '101_Anne Marie - 2002 .mp3'],
            ['mood' => 'happy', 'title' => 'Baby', 'artist' => 'Justin Bieber feat. Ludacris', 'lyrics' => 'Baby lyrics...', 'filename' => '102_Baby - Justin Bieber feat. Ludacris.mp3'],
            ['mood' => 'happy', 'title' => 'Just The Way You Are', 'artist' => 'Bruno Mars', 'lyrics' => 'Just The Way You Are lyrics...', 'filename' => '103_Bruno Mars - Just The Way You Are.mp3'],
            ['mood' => 'happy', 'title' => 'Sit Still, Look Pretty', 'artist' => 'Daya', 'lyrics' => 'Sit Still, Look Pretty lyrics...', 'filename' => '104_Daya - Sit Still, Look Pretty.mp3'],
            ['mood' => 'happy', 'title' => 'Domino', 'artist' => 'Jessie J', 'lyrics' => 'Domino lyrics...', 'filename' => '105_Domino - Jessie J.mp3'],
            ['mood' => 'happy', 'title' => 'Love Me Like You Do', 'artist' => 'Ellie Goulding', 'lyrics' => 'Love Me Like You Do lyrics...', 'filename' => '106_Ellie Goulding - Love Me Like You Do.mp3'],
            ['mood' => 'happy', 'title' => 'Thats So True', 'artist' => 'Gracie Abrams', 'lyrics' => 'Thats So True lyrics...', 'filename' => '107_Gracie Abrams - Thats So True.mp3'],
            ['mood' => 'happy', 'title' => 'On The Floor', 'artist' => 'Jennifer Lopez ft. Pitbull', 'lyrics' => 'On The Floor lyrics...', 'filename' => '108_Jennifer Lopez - On The Floor ft. Pitbull.mp3'],
            ['mood' => 'happy', 'title' => 'Price Tag', 'artist' => 'Jessie J Feat. B.O.B', 'lyrics' => 'Price Tag lyrics...', 'filename' => '109_Jessie J - Price Tag Feat. B.O.B.mp3'],
            ['mood' => 'happy', 'title' => 'Last Friday Night', 'artist' => 'Katy Perry', 'lyrics' => 'Last Friday Night lyrics...', 'filename' => '110_Katy Perry - Last Friday Night.mp3'],
            ['mood' => 'happy', 'title' => 'Sugar', 'artist' => 'Maroon 5', 'lyrics' => 'Sugar lyrics...', 'filename' => '111_Maroon 5 - Sugar.mp3'],
            ['mood' => 'happy', 'title' => 'Payphone', 'artist' => 'Maroon 5, Wiz Khalifa', 'lyrics' => 'Payphone lyrics...', 'filename' => '112_Maroon 5, Wiz Khalifa  Payphone.mp3'],
            ['mood' => 'happy', 'title' => 'One Time', 'artist' => 'Justin Bieber', 'lyrics' => 'One Time lyrics...', 'filename' => '113_One Time - Justin Bieber.mp3'],
            ['mood' => 'happy', 'title' => 'Good Time', 'artist' => 'Owl City, Carly Rae Jepsen', 'lyrics' => 'Good Time lyrics...', 'filename' => '114_Owl City, Carly Rae Jepsen - Good Time.mp3'],
            ['mood' => 'happy', 'title' => 'You Da One', 'artist' => 'Rihanna', 'lyrics' => 'You Da One lyrics...', 'filename' => '115_Rihanna - You Da One.mp3'],
            ['mood' => 'happy', 'title' => 'Eenie Meenie', 'artist' => 'Sean Kingston, Justin Bieber', 'lyrics' => 'Eenie Meenie lyrics...', 'filename' => '116_Sean Kingston, Justin Bieber - Eenie Meenie.mp3'],
            ['mood' => 'happy', 'title' => 'Stitches', 'artist' => 'Shawn Mendes', 'lyrics' => 'Stitches lyrics...', 'filename' => '117_Shawn Mendes - Stitches.mp3'],
            ['mood' => 'happy', 'title' => 'Treat You Better', 'artist' => 'Shawn Mendes', 'lyrics' => 'Treat You Better lyrics...', 'filename' => '118_Shawn Mendes - Treat You Better.mp3'],
            ['mood' => 'happy', 'title' => 'Somebody To You', 'artist' => 'Various Artists', 'lyrics' => 'Somebody To You lyrics...', 'filename' => '119_Somebody To You.mp3'],
            ['mood' => 'happy', 'title' => 'Girl In The Miror', 'artist' => 'Sophia Grace ft. Silento', 'lyrics' => 'Girl In The Miror lyrics...', 'filename' => '120_Sophia Grace - Girl In The Miror ft. Silento.mp3'],
            ['mood' => 'happy', 'title' => 'supernatural', 'artist' => 'Ariana Grande', 'lyrics' => 'supernatural lyrics...', 'filename' => '121_supernatural - Ariana Grande.mp3'],
            // REMOVED: Taylor Swift - 22
            // REMOVED: Taylor Swift - Fearless
            ['mood' => 'happy', 'title' => 'Paper Rings', 'artist' => 'Taylor Swift', 'lyrics' => 'Paper Rings lyrics...', 'filename' => '124_Taylor Swift - Paper Rings.mp3'],
            ['mood' => 'happy', 'title' => 'Every Summertime', 'artist' => 'NIKI', 'lyrics' => 'Every Summertime lyrics...', 'filename' => '125_NIKI - Every Summertime Every year we get older.mp3'],

            ['mood' => 'happy', 'title' => 'Ordinary', 'artist' => 'Alex Warren', 'lyrics' => 'Ordinary lyrics...', 'filename' => '126_Alex Warren - Ordinary (Official Video).mp3'],
            ['mood' => 'happy', 'title' => 'Carry You Home', 'artist' => 'Alex Warren', 'lyrics' => 'Carry You Home lyrics...', 'filename' => '127_Alex Warren - Carry You Home (Official Video).mp3'],
            ['mood' => 'happy', 'title' => 'The Nights', 'artist' => 'Avicii', 'lyrics' => 'The Nights lyrics...', 'filename' => '128_Avicii - The Nights.mp3'],
            ['mood' => 'happy', 'title' => 'Heart Attack', 'artist' => 'Demi Lovato', 'lyrics' => 'Heart Attack lyrics...', 'filename' => '129_Demi Lovato - Heart Attack (Official Video).mp3'],
            ['mood' => 'happy', 'title' => 'Dreams', 'artist' => 'DOLF & Weird Genius ft. Rochelle', 'lyrics' => 'Dreams lyrics...', 'filename' => '130_DOLF & Weird Genius - Dreams ft. Rochelle (Official Lyric Video).mp3'],
            ['mood' => 'happy', 'title' => 'Thunder', 'artist' => 'Imagine Dragons', 'lyrics' => 'Thunder lyrics...', 'filename' => '131_Imagine Dragons - Thunder (1).mp3'],
            ['mood' => 'happy', 'title' => 'Favorite Girl', 'artist' => 'Justin Bieber', 'lyrics' => 'Favorite Girl lyrics...', 'filename' => '132_Justin Bieber - Favorite Girl (Lyrics).mp3'],
            ['mood' => 'happy', 'title' => 'Sorry', 'artist' => 'Justin Bieber', 'lyrics' => 'Sorry lyrics...', 'filename' => '133_Justin Bieber - Sorry (Lyric Video).mp3'],
            ['mood' => 'happy', 'title' => 'dna', 'artist' => 'LANY', 'lyrics' => 'dna lyrics...', 'filename' => '134_LANY - dna (official lyric video).mp3'],
            ['mood' => 'happy', 'title' => 'Animals', 'artist' => 'Maroon 5', 'lyrics' => 'Animals lyrics...', 'filename' => '135_Maroon 5 - Animals [qpgTC9MDx1o] (1).mp3'],
            ['mood' => 'happy', 'title' => 'Daylight', 'artist' => 'Maroon 5', 'lyrics' => 'Daylight lyrics...', 'filename' => '136_Maroon 5 - Daylight (Official Music Video).mp3'],
            ['mood' => 'happy', 'title' => 'The Thing I Love', 'artist' => 'MAX & Andy Grammer', 'lyrics' => 'The Thing I Love lyrics...', 'filename' => '137_MAX & Andy Grammer - The Thing I Love (Official Video).mp3'],
            ['mood' => 'happy', 'title' => 'HONEY & LEMON', 'artist' => 'NAYKILLA', 'lyrics' => 'HONEY & LEMON lyrics...', 'filename' => '138_NAYKILLA - HONEY &  LEMON.mp3'],
            ['mood' => 'happy', 'title' => 'So Easy', 'artist' => 'Olivia Dean', 'lyrics' => 'So Easy lyrics...', 'filename' => '139_Olivia Dean - So Easy (To Fall In Love).mp3'],
            ['mood' => 'happy', 'title' => 'Steal My Girl', 'artist' => 'One Direction', 'lyrics' => 'Steal My Girl lyrics...', 'filename' => '140_One Direction - Steal My Girl.mp3'],
            ['mood' => 'happy', 'title' => 'Gone, Gone, Gone', 'artist' => 'Phillip Phillips', 'lyrics' => 'Gone, Gone, Gone lyrics...', 'filename' => '141_Phillip Phillips - Gone, Gone, Gone.mp3'],
            ['mood' => 'happy', 'title' => 'Locked Away', 'artist' => 'R. City ft. Adam Levine', 'lyrics' => 'Locked Away lyrics...', 'filename' => '142_R. City - Locked Away ft. Adam Levine.mp3'],
            ['mood' => 'happy', 'title' => 'THE SHADE', 'artist' => 'Rex Orange County', 'lyrics' => 'THE SHADE lyrics...', 'filename' => '143_Rex Orange County - THE SHADE (Official Audio).mp3'],
            ['mood' => 'happy', 'title' => 'I Gotta Feeling', 'artist' => 'The Black Eyed Peas', 'lyrics' => 'I Gotta Feeling lyrics...', 'filename' => '144_The Black Eyed Peas - I Gotta Feeling (Official Music Video).mp3'],
            ['mood' => 'happy', 'title' => 'Closer', 'artist' => 'The Chainsmokers ft. Halsey', 'lyrics' => 'Closer lyrics...', 'filename' => '145_The Chainsmokers - Closer (Lyric) ft. Halsey.mp3'],
            ['mood' => 'happy', 'title' => 'Blinding Lights', 'artist' => 'The Weeknd', 'lyrics' => 'Blinding Lights lyrics...', 'filename' => '146_The Weeknd - Blinding Lights (Official Video).mp3'],
            ['mood' => 'happy', 'title' => 'Sweet Scar', 'artist' => 'Weird Genius ft. Prince Husein', 'lyrics' => 'Sweet Scar lyrics...', 'filename' => '147_Weird Genius - Sweet Scar (ft. Prince Husein) Official Music Video.mp3'],
            ['mood' => 'happy', 'title' => 'Till It Hurts', 'artist' => 'Yellow Claw ft. Ayden', 'lyrics' => 'Till It Hurts lyrics...', 'filename' => '148_Yellow Claw - Till It Hurts ft. Ayden [Official Music Video].mp3'],
            ['mood' => 'happy', 'title' => 'Stay', 'artist' => 'Zedd, Alessia Cara', 'lyrics' => 'Stay lyrics...', 'filename' => '149_Zedd, Alessia Cara - Stay (Official Music Video).mp3'],
            ['mood' => 'happy', 'title' => 'The Middle', 'artist' => 'Zedd, Maren Morris, Grey', 'lyrics' => 'The Middle lyrics...', 'filename' => '150_Zedd, Maren Morris, Grey - The Middle (Official Music Video).mp3'],

            // SAD SONGS (49 songs - removed 26 missing)
            ['mood' => 'sad', 'title' => 'Hello', 'artist' => 'Adele', 'lyrics' => 'Hello lyrics...', 'filename' => '151_Adele - Hello (Official Music Video).mp3'],
            ['mood' => 'sad', 'title' => 'All Too Well', 'artist' => 'Taylor Swift', 'lyrics' => 'All Too Well lyrics...', 'filename' => '152_All Too Well (10 Minute Version) (Taylor\'s Version) (From The Vault) (Lyric Video).mp3'],
            ['mood' => 'sad', 'title' => 'Talking To The Moon', 'artist' => 'Bruno Mars', 'lyrics' => 'Talking To The Moon lyrics...', 'filename' => '153_Bruno Mars - Talking To The Moon (Official Lyric Video).mp3'],
            ['mood' => 'sad', 'title' => 'When I Was Your Man', 'artist' => 'Bruno Mars', 'lyrics' => 'When I Was Your Man lyrics...', 'filename' => '154_Bruno Mars - When I Was Your Man (Official Music Video).mp3'],
            ['mood' => 'sad', 'title' => 'A Thousand Years', 'artist' => 'Christina Perri', 'lyrics' => 'A Thousand Years lyrics...', 'filename' => '155_Christina Perri - A Thousand Years [Official Music Video].mp3'],
            ['mood' => 'sad', 'title' => 'Thinking Out Loud', 'artist' => 'Ed Sheeran', 'lyrics' => 'Thinking Out Loud lyrics...', 'filename' => '156_Ed Sheeran  Thinking Out Loud  Lirik Lagu Terjemahan.mp3'],
            ['mood' => 'sad', 'title' => 'Perfect', 'artist' => 'Ed Sheeran', 'lyrics' => 'Perfect lyrics...', 'filename' => '157_Ed Sheeran - Perfect (Official Music Video).mp3'],
            ['mood' => 'sad', 'title' => 'Photograph', 'artist' => 'Ed Sheeran', 'lyrics' => 'Photograph lyrics...', 'filename' => '158_Ed Sheeran - Photograph (Official Music Video).mp3'],
            ['mood' => 'sad', 'title' => 'Sign of the Times', 'artist' => 'Harry Styles', 'lyrics' => 'Sign of the Times lyrics...', 'filename' => '159_Harry Styles - Sign of the Times (Official Video).mp3'],
            ['mood' => 'sad', 'title' => 'Demons', 'artist' => 'Imagine Dragons', 'lyrics' => 'Demons lyrics...', 'filename' => '160_Imagine Dragons - Demons (Official Music Video).mp3'],
            ['mood' => 'sad', 'title' => 'Shallow', 'artist' => 'Lady Gaga, Bradley Cooper', 'lyrics' => 'Shallow lyrics...', 'filename' => '161_Lady Gaga, Bradley Cooper - Shallow (from A Star Is Born) (Official Music Video).mp3'],
            ['mood' => 'sad', 'title' => 'Someone You Loved', 'artist' => 'Lewis Capaldi', 'lyrics' => 'Someone You Loved lyrics...', 'filename' => '162_Lewis Capaldi - Someone You Loved.mp3'],
            ['mood' => 'sad', 'title' => 'Tattoo', 'artist' => 'Loreen', 'lyrics' => 'Tattoo lyrics...', 'filename' => '163_Loreen - Tattoo.mp3'],
            ['mood' => 'sad', 'title' => 'Love Someone', 'artist' => 'Lukas Graham', 'lyrics' => 'Love Someone lyrics...', 'filename' => '164_Lukas Graham - Love Someone [Official Music Video].mp3'],
            ['mood' => 'sad', 'title' => 'Memories', 'artist' => 'Maroon 5', 'lyrics' => 'Memories lyrics...', 'filename' => '165_Maroon 5 - Memories (Official Video).mp3'],
            ['mood' => 'sad', 'title' => 'This Town', 'artist' => 'Niall Horan', 'lyrics' => 'This Town lyrics...', 'filename' => '166_Niall Horan - This Town (Official Lyric Video).mp3'],
            ['mood' => 'sad', 'title' => 'happier', 'artist' => 'Olivia Rodrigo', 'lyrics' => 'happier lyrics...', 'filename' => '167_Olivia Rodrigo - happier (Lyric Video).mp3'],
            // REMOVED: All remaining sad songs from 168-200 (One-Direction-Story-of-My-Life through øneheart x reidenshi - Snowfall)
        ];

        // Function to encode filename for Azure path
        function encodeAzurePath($filename) {
            // Remove extension (.wav or .mp3)
            $name = pathinfo($filename, PATHINFO_FILENAME);
            
            // Remove the ID prefix (e.g., "118_" from "118_Shawn Mendes...")
            if (preg_match('/^\d+_(.+)$/', $name, $matches)) {
                $name = $matches[1];
            }
            
            // Only replace spaces with %20, keep everything else as-is
            $encodedName = str_replace(' ', '%20', $name);
            
            // Return with songs/ prefix and .mp3 extension
            return 'songs/' . $encodedName . '.mp3';
        }

        // Insert songs into database
        foreach ($songs as $index => $song) {
            $encodedPath = encodeAzurePath($song['filename']);
            
            Songs::create([
                'userId' => $userId,
                'moodId' => $moodIds[$song['mood']],
                'confidence' => rand(70, 100), // Random confidence between 70-100
                'title' => $song['title'],
                'lyrics' => $song['lyrics'],
                'artist' => $song['artist'],
                'genre' => 'Pop', // Default genre, adjust as needed
                'duration' => rand(120, 300), // Random duration between 2-5 minutes (in seconds)
                'publisher' => 'Various Publishers',
                'datePublished' => now()->subDays(rand(1, 365)),
                'songPath' => $encodedPath,
                'photoPath' => null, // Default photo path
            ]);
        }
    }
}